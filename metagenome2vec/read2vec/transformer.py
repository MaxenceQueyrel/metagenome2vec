import random
import dill
from pathlib import Path
import shutil
import os
import math
import time
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer
import torch.optim as optim
from torchtext import vocab
from torchtext.legacy.data import ReversibleField, Dataset
from torchtext.legacy.datasets import LanguageModelingDataset
from torch.nn import AdaptiveLogSoftmaxWithLoss
from torch.nn import Linear, Dropout, LayerNorm, TransformerEncoder
from torch.utils.data import DataLoader

from metagenome2vec.utils import parser_creator
from metagenome2vec.read2vec.read2vec import Read2Vec

from torchtext.nn import MultiheadAttentionContainer, InProjContainer, ScaledDotProduct


SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        S, N = x.size()
        pos = torch.arange(S,
                           dtype=torch.long,
                           device=x.device).unsqueeze(0).expand((N, S)).t()
        return self.pos_embedding(pos)


class TokenTypeEncoding(nn.Module):
    def __init__(self, type_token_num, d_model):
        super(TokenTypeEncoding, self).__init__()
        self.token_type_embeddings = nn.Embedding(type_token_num, d_model)

    def forward(self, seq_input, token_type_input):
        S, N = seq_input.size()
        if token_type_input is None:
            token_type_input = torch.zeros((S, N),
                                           dtype=torch.long,
                                           device=seq_input.device)
        return self.token_type_embeddings(token_type_input)


class BertEmbedding(nn.Module):
    def __init__(self, ntoken, ninp, dropout=0.5):
        super(BertEmbedding, self).__init__()
        self.ninp = ninp
        self.ntoken = ntoken
        self.pos_embed = PositionalEncoding(ninp)
        self.embed = nn.Embedding(ntoken, ninp)
        self.tok_type_embed = TokenTypeEncoding(2, ninp)  # Two sentence type
        self.norm = LayerNorm(ninp)
        self.dropout = Dropout(dropout)

    def forward(self, src, token_type_input):
        src = self.embed(src) + self.pos_embed(src) \
            + self.tok_type_embed(src, token_type_input)
        return self.dropout(self.norm(src))


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048,
                 dropout=0.1, activation="gelu"):
        super(TransformerEncoderLayer, self).__init__()
        in_proj_container = InProjContainer(Linear(d_model, d_model),
                                            Linear(d_model, d_model),
                                            Linear(d_model, d_model))
        self.mha = MultiheadAttentionContainer(nhead, in_proj_container,
                                               ScaledDotProduct(), Linear(d_model, d_model))
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            raise RuntimeError("only relu/gelu are supported, not {}".format(activation))

    def init_weights(self):
        self.mha.in_proj_container.query_proj.init_weights()
        self.mha.in_proj_container.key_proj.init_weights()
        self.mha.in_proj_container.value_proj.init_weights()
        self.mha.out_proj.init_weights()
        self.linear1.weight.data.normal_(mean=0.0, std=0.02)
        self.linear2.weight.data.normal_(mean=0.0, std=0.02)
        self.norm1.bias.data.zero_()
        self.norm1.weight.data.fill_(1.0)
        self.norm2.bias.data.zero_()
        self.norm2.weight.data.fill_(1.0)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        attn_output, attn_output_weights = self.mha(src, src, src, attn_mask=src_mask)
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class BertModel(nn.Module):
    """Contain a transformer encoder."""

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(BertModel, self).__init__()
        self.model_type = 'Transformer'
        self.bert_embed = BertEmbedding(ntoken, ninp)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.ninp = ninp

    def forward(self, src, token_type_input):
        src = self.bert_embed(src, token_type_input)
        output = self.transformer_encoder(src)
        return output


class MLMTask(nn.Module):
    """Contain a transformer encoder plus MLM head."""

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(MLMTask, self).__init__()
        self.bert_model = BertModel(ntoken, ninp, nhead, nhid, nlayers, dropout=0.5)
        self.mlm_span = Linear(ninp, ninp)
        self.activation = F.gelu
        self.norm_layer = LayerNorm(ninp, eps=1e-12)
        self.mlm_head = Linear(ninp, ntoken)

    def forward(self, src, token_type_input=None):
        src = src.transpose(0, 1)  # Wrap up by nn.DataParallel
        output = self.bert_model(src, token_type_input)
        output = self.mlm_span(output)
        output = self.activation(output)
        output = self.norm_layer(output)
        output = self.mlm_head(output)
        return output


class Seq2Seq(Read2Vec):
    def __init__(self, k_size, max_length=20, learning_rate=0.1,
                 id_gpu=[-1], batch_size=64, hid_dim=256, dropout=0.2,
                 n_iterations=-1, emb_dim=None, language_modeling=False,
                 nhead=6, nlayers=3, f_name="transformer", clip=0.1, log_interval=100):
        """
        TODO
        :param embedding: 2-D Numpy array, matrix of embeddings
        :param dico_index: dictionary with the kmer and its index in the embeddings matrix
        :param reverse_index: dictionary with index as key and kmer as value
        :param k_size: int, kmer size
        :param step: int, The number of nucleotides that separate each k_mer
                        if step == k_size there there is only one cut
        """
        Read2Vec.__init__(self, None, None, k_size)
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.hid_dim = hid_dim
        self.id_gpu = id_gpu
        self.device = torch.device("cuda:%s" % id_gpu[0] if id_gpu[0] != -1 else "cpu")
        # To be initialized with functions
        self.TEXT = None
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.cutoffs = None
        self.div_value = 2
        self.dropout = dropout
        self.n_iterations = n_iterations
        self.emb_dim = emb_dim
        self.nlayers = nlayers
        self.nhead = nhead
        self.ntokens = None
        self.mask_frac = 0.15
        self.index_pad = 1
        self.index_msk = 4
        self.language_modeling = language_modeling
        self.f_name = f_name
        self.log_interval = log_interval
        self.clip = clip

    @staticmethod
    def tokenize(x):
        return x.split()

    def process_raw_data(self, raw_data):
        _num = raw_data.size(0) // (self.batch_size * self.max_length)
        raw_data = raw_data[:(_num * self.batch_size * self.max_length)]
        return raw_data

    def collate_batch(self, batch_data):
        batch_data = torch.tensor(batch_data).long().view(batch_size, -1).t().contiguous()
        # Generate masks with args.mask_frac
        data_len = batch_data.size(0)
        ones_num = int(data_len * self.mask_frac)
        zeros_num = data_len - ones_num
        lm_mask = torch.cat([torch.zeros(zeros_num), torch.ones(ones_num)])
        lm_mask = lm_mask[torch.randperm(data_len)]
        batch_data = torch.cat((torch.tensor([[self.cls_id] * batch_data.size(1)]).long(), batch_data))
        lm_mask = torch.cat((torch.tensor([0.0]), lm_mask))
        targets = torch.stack([batch_data[i] for i in range(lm_mask.size(0)) if lm_mask[i]]).view(-1)
        batch_data = batch_data.masked_fill(lm_mask.bool().unsqueeze(1), self.mask_id)
        return batch_data, lm_mask, targets

    def create_field_iterator(self, path_data_train, path_data_valid, path_kmer2vec, nb_cutoffs=None):
        """
        Create the field and the iterator
        :param path_data_train: String, path of the training data
        :param path_data_valid: String, path of the validation data
        :param path_kmer2vec: String, path where is save the embeddings
        :return:
        """
        # Allow to get the minimum frequency to construct the torchtext vocab
        """
        if path_kmer_count is not None:
            reader = csv.reader(open(path_kmer_count, 'r'))
            min_freq = int(next((x for i, x in enumerate(reader) if i == max_vectors), None)[-1])
        else:
            min_freq = 1
        """
        # Load a pre-trained kmer model
        if path_kmer2vec is not None:
            vectors = vocab.Vectors(os.path.join(path_kmer2vec, "embeddings.csv"),
                                    cache="./cache_k%s" % k, max_vectors=max_vectors)
            if "</s>" in vectors.stoi:
                vectors.stoi["<eos>"] = vectors.stoi.pop("</s>")
        else:
            vectors = None

        self.TEXT = ReversibleField(tokenize=self.tokenize,
                                    init_token='<sos>',
                                    eos_token='<eos>',
                                    unk_token="<unk>",
                                    lower=False)

        self.train_data, fields, self.cutoffs = self.check_exist_and_load_dataset(path_data_train, with_fields=True)
        self.valid_data = self.check_exist_and_load_dataset(path_data_valid)

        if self.train_data is False or self.valid_data is False:
            self.train_data = LanguageModelingDataset(path_data_train, self.TEXT, newline_eos=True)
            self.valid_data = LanguageModelingDataset(path_data_valid, self.TEXT, newline_eos=True)

            self.TEXT.build_vocab(self.train_data, max_size=max_vectors, vectors=vectors,
                                  specials=['<unk>', '<pad>', '<MASK>'])
            #old_stoi = self.TEXT.vocab.stoi.copy()  # To retrieve good index in vectors matrix after changing stoi and itos
            #stoi, itos, cutoffs = self.init_stoi_itos_cutoffs([path_data_train], min_freq=min_freq,
            #                                                  nb_cutoffs=nb_cutoffs)
            #if self.TEXT.vocab.vectors is not None:
            #    self.TEXT.vocab.vectors = self.TEXT.vocab.vectors[np.array([old_stoi[x] for x in itos])]
            #del old_stoi
            #self.TEXT.vocab.stoi = stoi
            #self.TEXT.vocab.itos = itos
            self.cutoffs = None  #cutoffs

            self.remove_exist_and_save_dataset(self.train_data, path_data_train, self.cutoffs)
            self.remove_exist_and_save_dataset(self.valid_data, path_data_valid)
        else:
            self.TEXT = fields["text"]
        self.mask_id = self.TEXT.vocab.stoi["<MASK>"]
        self.cls_id = self.TEXT.vocab.stoi["<cls>"]
        self.train_data = torch.tensor([self.TEXT.vocab.stoi[x] for x in self.train_data.examples[0].text])
        self.valid_data = torch.tensor([self.TEXT.vocab.stoi[x] for x in self.valid_data.examples[0].text])
        self.train_data = self.process_raw_data(self.train_data)
        self.valid_data = self.process_raw_data(self.valid_data)

    def save_dataset(self, dataset, path, cutoffs=None):
        if not isinstance(path, Path):
            path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(dataset.examples, path / "examples.pkl", pickle_module=dill)
        torch.save(dataset.fields, path / "fields.pkl", pickle_module=dill)
        torch.save(cutoffs, path / "cutoffs.pkl", pickle_module=dill)

    def load_dataset(self, path, with_fields=False):
        if not isinstance(path, Path):
            path = Path(path)
        examples = torch.load(path / "examples.pkl", pickle_module=dill)
        fields = torch.load(path / "fields.pkl", pickle_module=dill)
        cutoffs = torch.load(path / "cutoffs.pkl", pickle_module=dill)
        if with_fields:
            return Dataset(examples, fields), fields, cutoffs
        return Dataset(examples, fields)

    def remove_exist_and_save_dataset(self, dataset, path, cutoffs=None):
        if path.endswith("/"):
            path = path[:-1]
        f_name = self.f_name[:-3] if self.f_name.endswith(".pt") else self.f_name
        path_tmp = "_".join([path, f_name])
        if os.path.exists(path_tmp):
            os.rmdir(path_tmp)
        return self.save_dataset(dataset, path_tmp, cutoffs)

    def check_exist_and_load_dataset(self, path, with_fields=False):
        if path.endswith("/"):
            path = path[:-1]
        f_name = self.f_name[:-3] if self.f_name.endswith(".pt") else self.f_name
        path_tmp = "_".join([path, f_name])
        if overwrite:
            shutil.rmtree(path_tmp, ignore_errors=True)
        if not os.path.exists(path_tmp):
            if with_fields:
                return False, False, False
            return False
        return self.load_dataset(path_tmp, with_fields)

    def create_model(self):
        """
        Initialize vsariables : model, criterion and optimizer
        :param hid_dim: Int, size of the hidden dimension
        :return:
        """
        self.ntokens = len(self.TEXT.vocab)  # the size of vocabulary
        print(self.ntokens)
        self.emb_dim = self.TEXT.vocab.vectors.shape[1] if self.TEXT.vocab.vectors is not None else self.emb_dim
        assert self.emb_dim is not None, "You have to provide an ambeddings dimension or embeddings vectors"
        self.model = MLMTask(self.ntokens, self.emb_dim, self.nhead, self.hid_dim, self.nlayers, self.dropout).to(self.device)

        if self.id_gpu[0] != -1 and len(self.id_gpu) > 1:
            self.model = nn.DataParallel(self.model, device_ids=self.id_gpu).to(self.device)
            #self.model.module.apply(initialize_weights)
            if self.TEXT.vocab.vectors is not None:
                self.model.module.encoder.weight.data.copy_(self.TEXT.vocab.vectors)
        else:
            self.model = self.model.to(self.device)
            #self.model.apply(initialize_weights)
            if self.TEXT.vocab.vectors is not None:
                self.model.bert_model.bert_embed.embed.weight.data.copy_(self.TEXT.vocab.vectors)

        #self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1.0, gamma=0.1)
        if self.cutoffs is None:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = AdaptiveLogSoftmaxWithLoss(self.emb_dim,
                                                        self.ntokens,
                                                        cutoffs=self.cutoffs,
                                                        div_value=self.div_value).to(self.device)

    def train(self, train_loss_log, epoch):
        self.model.train()
        total_loss = 0.
        start_time = time.time()
        train_loss_log.append(0.0)
        dataloader = DataLoader(self.train_data, batch_size=self.batch_size * self.max_length,
                                shuffle=True,
                                collate_fn=lambda b: self.collate_batch(b))

        for batch, (data, lm_mask, targets) in enumerate(dataloader):
            if batch > self.n_iterations:
                break
            self.optimizer.zero_grad()
            data = data.to(self.device)
            targets = targets.to(self.device)
            data = data.transpose(0, 1)  # Wrap up by DDP or DataParallel
            output = self.model(data)
            output = torch.stack([output[i] for i in range(lm_mask.size(0)) if lm_mask[i]])
            loss = self.criterion(output.view(-1, self.ntokens), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            self.optimizer.step()
            total_loss += loss.item()
            if batch % self.log_interval == 0 and batch > 0:
                cur_loss = total_loss / self.log_interval
                elapsed = time.time() - start_time
                train_loss_log[-1] = cur_loss
                print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | '
                      'loss {:5.2f} | ppl {:8.2f}'.format(epoch, batch,
                                                          len(self.train_data) // (self.max_length * self.batch_size),
                                                          self.scheduler.get_last_lr()[0],
                                                          elapsed * 1000 / self.log_interval,
                                                          cur_loss, math.exp(cur_loss)))
                total_loss = 0
                start_time = time.time()
                print([self.TEXT.vocab.itos[x] for x in torch.argmax(F.softmax(self.model(data), dim=2)[:, 0, :], dim=1).cpu().numpy()])

    def evaluate(self):
        # Turn on evaluation mode which disables dropout.
        self.model.eval()
        total_loss = 0.
        dataloader = DataLoader(self.valid_data, batch_size=self.batch_size * self.max_length,
                                shuffle=True,
                                collate_fn=lambda b: self.collate_batch(b))
        with torch.no_grad():
            for batch, (data, lm_mask, targets) in enumerate(dataloader):
                if batch > self.n_iterations:
                    break
                data = data.to(self.device)
                targets = targets.to(self.device)
                data = data.transpose(0, 1)  # Wrap up by DDP or DataParallel
                output = self.model(data)
                output = torch.stack([output[i] for i in range(lm_mask.size(0)) if lm_mask[i]])
                output_flat = output.view(-1, self.ntokens)
                total_loss += self.criterion(output_flat, targets).item()
        return total_loss / ((len(self.valid_data) - 1) / self.max_length / batch_size)

    def fit(self, epochs, path_model):
        best_val_loss = None
        train_loss_log, val_loss_log = [], []
        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()
            self.train(train_loss_log, epoch)
            val_loss = self.evaluate()
            val_loss_log.append(val_loss)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                  'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                             val_loss, math.exp(val_loss)))
            print('-' * 89)
            if not best_val_loss or val_loss < best_val_loss:
                self.saveModel(path_model)
                best_val_loss = val_loss
            else:
                self.scheduler.step()

    def _transform(self, x):
        n_batch = 1 if len(x.shape) == 1 else x.shape[0]
        length = len(x) if len(x.shape) == 1 else x.shape[1]
        with torch.no_grad():
            tens = torch.tensor(x).view(n_batch, length).to(self.device)
            return torch.mean(self.model.bert_model(tens, None), axis=1).detach().cpu().numpy()

    def transform_all(self, X):
        """
        Transform a matrix of kmer (one line a read, one column a kmer) into a matrix of reads embeddings
        => one line a read one column a dimension of an embedding
        :param X: Numpy 2-D array, index of the kmer in reverse_index (int to kmer)
        :return: Numpy 2-D array
        """
        with torch.no_grad():
            X = np.array_split(X, 1 + len(X) // self.batch_size)
            L_res = []
            for x_batch in X:
                L_res.append(self._transform(x_batch))
            return np.concatenate(L_res)

    def loadModel(self, path_model, device=None):
        """
        :param path_model: String, The complete path where is saved the model
        :return:
        """
        assert path_model[-3:] == ".pt", "The file name should end by .pt (.pytorch)"
        self.dico_index = torch.load(path_model.replace(".pt", "-field.pt"))
        if "<unk>" in self.dico_index:
            self.index_unk = self.dico_index["<unk>"]
        if "<pad>" in self.dico_index:
            self.index_pad = self.dico_index["<pad>"]
        else:
            self.index_pad = 1
        parameters = torch.load(path_model.replace(".pt", "-parameters.pt"))
        self.hid_dim = parameters["hid_dim"]
        self.dropout = parameters["dropout"]
        self.cutoffs = parameters["cutoffs"]
        self.div_value = parameters["div_value"]
        self.max_length = parameters["max_length"]
        self.emb_dim = parameters["emb_dim"]
        self.ntokens = parameters["ntokens"]
        self.nhead = parameters["nhead"]
        self.nlayers = parameters["nlayers"]
        self.language_modeling = parameters["language_modeling"]
        self.clip = parameters["clip"]
        self.log_interval = parameters["log_interval"]
        if device is not None:
            self.device = device
        else:
            self.device = parameters["device"] if self.device is None else self.device
        self.model = MLMTask(self.ntokens, self.emb_dim, self.nhead, self.hid_dim, self.nlayers, self.dropout).to(self.device)
        parameters = torch.load(path_model, map_location=self.device)
        if self.id_gpu[0] != -1 and len(self.id_gpu) > 1:
            self.model = nn.DataParallel(self.model, device_ids=self.id_gpu).to(self.device)
            parameters = {"module." + k if k[:7] != "module." else k: v for k, v in parameters.items()}
        else:
            # In order to key match if the model is saved as DataParallel
            parameters = {k.replace("module.", ""): v for k, v in parameters.items()}
        self.model.load_state_dict(parameters)

    def saveModel(self, path_model):
        """
        Save the seq2seq model
        :param path_model: str, complete path name of the model
        :return:
        """
        assert path_model[-3:] == ".pt", "The file name should end by .pt (.pytorch)"

        torch.save(self.model.state_dict(), path_model)
        torch.save(dict(self.TEXT.vocab.stoi), path_model.replace(".pt", "-field.pt"))
        parameters = {"device": self.device,
                      "hid_dim": self.hid_dim,
                      "dropout": self.dropout,
                      "cutoffs": self.cutoffs,
                      "div_value": self.div_value,
                      "max_length": self.max_length,
                      "emb_dim": self.emb_dim,
                      "ntokens": self.ntokens,
                      "nhead": self.nhead,
                      "nlayers": self.nlayers,
                      "language_modeling": self.language_modeling,
                      "log_interval": self.log_interval,
                      "clip": self.clip}
        torch.save(parameters, path_model.replace(".pt", "-parameters.pt"))


if __name__ == "__main__":
    parser = parser_creator.ParserCreator()
    args = parser.parser_transformer()

    k = args.k_mer_size
    f_name = args.f_name
    f_name += ".pt" if not f_name.endswith('.pt') else ""
    n_steps = args.n_steps
    n_iterations = args.n_iterations
    batch_size = args.batch_size
    id_gpu = [int(x) for x in args.id_gpu.split(',')]
    n_cpus = args.n_cpus
    torch.set_num_threads(n_cpus)
    learning_rate = args.learning_rate
    emb_dim = args.embedding_size
    hid_dim = args.hidden_size
    nb_cutoffs = args.nb_cutoffs
    dropout = args.dropout  # ref 0.1
    path_kmer2vec = args.path_kmer2vec
    language_modeling = args.language_modeling
    nhead = args.nhead
    nlayers = args.nlayers
    path_kmer_count = args.path_kmer_count
    max_vectors = args.max_vectors
    overwrite = args.overwrite
    clip = args.clip

    path_analysis = args.path_analysis
    path_data_train, path_data_valid = args.path_data.split(',')
    max_length = args.max_length

    path_log = args.path_log
    log_file = args.log_file

    log = logger.Logger(path_log, log_file, log_file,
                        variable_time={"k": k, "embedding_size": emb_dim,
                                       "n_steps": n_steps, "batch_size": batch_size},
                        **vars(args))

    print("Initializing seq2seq model")
    model = Seq2Seq(k, max_length=max_length, learning_rate=learning_rate, id_gpu=id_gpu, batch_size=batch_size,
                    hid_dim=hid_dim,  dropout=dropout, n_iterations=n_iterations, emb_dim=emb_dim,
                    nhead=nhead, nlayers=nlayers, f_name=f_name, clip=clip)
    print("Creating field iterator")
    model.create_field_iterator(path_data_train, path_data_valid, path_kmer2vec, nb_cutoffs)
    print("Creating Model")
    model.create_model()
    hdfs.create_dir(os.path.join(path_analysis, "read2vec"), mode="local")
    print("Fit Model")
    model.fit(epochs=n_steps, path_model=os.path.join(path_analysis, "read2vec", f_name))





