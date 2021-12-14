from read2genome import Read2Genome
import os
import sys
import math
import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pyspark.sql import functions as F
from pyspark.sql import types as T
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from packaging import version
import torchtext
if version.parse("0.9.0") <= version.parse(torchtext.__version__):
    del torchtext
    import torchtext.legacy as torchtext
    from torchtext.legacy import vocab
    from torchtext.legacy.data import Field, LabelField
    from torchtext.legacy.data import TabularDataset, BucketIterator, Dataset
else:
    from torchtext import vocab
    from torchtext.data import Field, LabelField
    from torchtext.data import TabularDataset, BucketIterator, Dataset

import torch.optim as optim
import dill
from pathlib import Path


root_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, os.path.join(root_folder, "utils"))
sys.path.insert(0, os.path.join(root_folder, "read2vec"))
sys.path.insert(0, os.path.join(root_folder, "read2genome"))

from read2vec import Read2Vec
import parser_creator
import logger
import hdfs_functions as hdfs

if sys.version_info[0] == 3 and sys.version_info[1] == 7:
    import transformation_ADN
else:
    import transformation_ADN2 as transformation_ADN

#os.environ['ARROW_PRE_0_15_IPC_FORMAT'] = '1'


class TransformerModel(nn.Module):
    def __init__(self, ntoken, nclass, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, nclass)
        self.init_weights()

    def forward(self, src, get_embeddings=False):
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        if get_embeddings:
            return output
        return torch.mean(self.decoder(output), axis=0)

    def init_weights(self):
        initrange = 0.1
        nn.init.xavier_uniform_(self.encoder.weight.data)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


class TransformerClassifier(Read2Genome, Read2Vec):
    def __init__(self, k_size, max_length=20, learning_rate=0.001,
                 id_gpu=[-1], batch_size=64, hid_dim=256, dropout=0.1,
                 n_iterations=-1, emb_dim=None, nhead=6, nlayers=3, tax_level="tax_id",
                 f_name="transformer_classifier"):
        Read2Genome.__init__(self, "transformer_classifier")
        Read2Vec.__init__(self, None, None, k_size, max_length)
        self.k_size = k_size
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.hid_dim = hid_dim
        self.id_gpu = id_gpu
        self.device = torch.device("cuda:%s" % id_gpu[0] if id_gpu[0] != -1 else "cpu")
        # To be initialized with functions
        self.TEXT = None
        self.LABEL = None
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.dropout = dropout
        self.n_iterations = n_iterations
        self.emb_dim = emb_dim
        self.nlayers = nlayers
        self.nhead = nhead
        self.ntokens = None
        self.p_remove = 0.15
        self.index_pad = 1
        self.index_msk = 4
        self.tax_level = tax_level
        self.f_name = f_name

    def tokeinze_wrapper(self):
        k = self.k_size
        def tokenize(text):
            """
            Tokenizes English text from a string into a list of strings (tokens)
            """
            return transformation_ADN.cut_word(text, k)
        return tokenize

    def save_dataset(self, dataset, path):
        if not isinstance(path, Path):
            path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(dataset.examples, path / "examples.pkl", pickle_module=dill)
        torch.save(dataset.fields, path / "fields.pkl", pickle_module=dill)

    def load_dataset(self, path, with_fields=False):
        if not isinstance(path, Path):
            path = Path(path)
        examples = torch.load(path / "examples.pkl", pickle_module=dill)
        fields = torch.load(path / "fields.pkl", pickle_module=dill)
        if with_fields:
            return Dataset(examples, fields), fields
        return Dataset(examples, fields)

    def remove_exist_and_save_dataset(self, dataset, path):
        if path.endswith("/"):
            path = path[:-1]
        f_name = self.f_name[:-3] if self.f_name.endswith(".pt") else self.f_name
        path_tmp = "_".join([path, f_name])
        if os.path.exists(path_tmp):
            os.rmdir(path_tmp)
        return self.save_dataset(dataset, path_tmp)

    def check_exist_and_load_dataset(self, path, with_fields=False):
        if path.endswith("/"):
            path = path[:-1]
        f_name = self.f_name[:-3] if self.f_name.endswith(".pt") else self.f_name
        path_tmp = "_".join([path, f_name])
        if not os.path.exists(path_tmp):
            if with_fields:
                return False, False
            return False
        return self.load_dataset(path_tmp, with_fields)

    def create_field_iterator(self, path_data_train, path_data_valid, path_kmer2vec):
        """
        Create the field and the iterator
        :param path_data_train: String, path of the training data
        :param path_data_valid: String, path of the validation data
        :param path_kmer2vec: String, path where is save the embeddings
        :return:
        """
        self.TEXT = Field(tokenize=self.tokeinze_wrapper(),
                          init_token='<sos>',
                          eos_token='<eos>',
                          unk_token="<unk>",
                          pad_token='<pad>',
                          fix_length=self.max_length,
                          lower=False)
        self.LABEL = LabelField(dtype=torch.long)

        if path_kmer2vec is not None:
            vectors = vocab.Vectors(os.path.join(path_kmer2vec, "embeddings.csv"), cache="./cache_k%s" % self.k_size)
            if "</s>" in vectors.stoi:
                vectors.stoi["<eos>"] = vectors.stoi.pop("</s>")
        else:
            vectors = None
        min_freq = 1

        def default():
            return 0

        self.train_data, fields = self.check_exist_and_load_dataset(path_data_train, with_fields=True)
        self.valid_data = self.check_exist_and_load_dataset(path_data_valid)

        if self.train_data is False or self.valid_data is False:
            fields = [('text', self.TEXT), ('label', self.LABEL)]
            self.train_data = TabularDataset(path_data_train, format='TSV', fields=fields, skip_header=True)
            self.valid_data = TabularDataset(path_data_valid, format='TSV', fields=fields, skip_header=True)

            self.TEXT.build_vocab(self.train_data, min_freq=min_freq, vectors=vectors)
            self.TEXT.vocab.stoi.default_factory = default
            self.LABEL.build_vocab(self.train_data, min_freq=min_freq)

            self.remove_exist_and_save_dataset(self.train_data, path_data_train)
            self.remove_exist_and_save_dataset(self.valid_data, path_data_valid)
        else:
            self.TEXT, self.LABEL = fields["text"], fields["label"]

        self.train_data, self.valid_data = BucketIterator.splits(
            (self.train_data, self.valid_data),
            batch_size=self.batch_size,
            device=self.device,
            sort=False,
            repeat=False,
            shuffle=True)

    def create_model(self):
        """
        Initialize vsariables : model, criterion and optimizer
        :param hid_dim: Int, size of the hidden dimension
        :return:
        """
        self.nclass = len(self.LABEL.vocab.stoi)  # the size of vocabulary
        self.ntokens = len(self.TEXT.vocab.stoi)  # the size of vocabulary
        self.emb_dim = self.TEXT.vocab.vectors.shape[1] if self.TEXT.vocab.vectors is not None else self.emb_dim
        assert self.emb_dim is not None, "You have to provide an ambeddings dimension or embeddings vectors"
        self.model = TransformerModel(self.ntokens, self.nclass, self.emb_dim, self.nhead,
                                      self.hid_dim, self.nlayers, self.dropout).to(self.device)
        self.model.init_weights()

        if self.id_gpu[0] != -1 and len(self.id_gpu) > 1:
            self.model = nn.DataParallel(self.model, device_ids=self.id_gpu).to(self.device)
            #self.model.module.apply(initialize_weights)
            if self.TEXT.vocab.vectors is not None:
                self.model.module.encoder.weight.data.copy_(self.TEXT.vocab.vectors)
        else:
            self.model = self.model.to(self.device)
            #self.model.apply(initialize_weights)
            if self.TEXT.vocab.vectors is not None:
                self.model.encoder.weight.data.copy_(self.TEXT.vocab.vectors)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1.0, gamma=0.95)
        self.criterion = nn.CrossEntropyLoss()

    def train(self, clip):
        self.model.train()  # Turn on the train mode
        total_loss = 0.
        inter_loss = 0.
        if self.n_iterations > 0:
            n_iterations = min(self.n_iterations, len(self.train_data))
        else:
            n_iterations = len(self.train_data)
        log_interval = n_iterations // 10
        for i, batch in enumerate(iter(self.train_data)):
            if i > n_iterations:
                break
            start_time = time.time()
            self.optimizer.zero_grad()
            data, targets = batch.text, batch.label
            output = self.model(data).squeeze()
            loss = self.criterion(output, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
            self.optimizer.step()

            total_loss += loss.detach().cpu().item()
            inter_loss += loss.detach().cpu().item()
            if i % log_interval == 0 and i > 0:
                cur_loss = inter_loss / log_interval
                elapsed = time.time() - start_time
                log.write('| {:5d}/{:5d} batches | '
                          'lr {:02.4f} | ms/batch {:5.2f} | '
                          'loss {:5.2f} | ppl {:8.2f}'.format(
                    i, len(self.train_data), self.scheduler.get_last_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss)))
                inter_loss = 0
        return total_loss / i

    def evaluate(self):
        self.model.eval()  # Turn on the evaluation mode
        total_loss = 0.
        with torch.no_grad():
            if self.n_iterations > 0:
                n_iterations = min(self.n_iterations, len(self.valid_data))
            else:
                n_iterations = len(self.valid_data) + 1
            for i, batch in enumerate(iter(self.valid_data)):
                if i > n_iterations:
                    break
                data, targets = batch.text, batch.label
                output = self.model(data).squeeze()
                loss = self.criterion(output, targets)
                loss = loss[1] if isinstance(loss, tuple) else loss
                total_loss += loss.detach().cpu().item()
        return total_loss / i

    def fit(self, epochs, clip, path_model):
        best_valid_loss = float("inf")
        for epoch in range(1, epochs + 1):
            start_time = time.time()

            train_loss = self.train(clip)
            valid_loss = self.evaluate()

            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            log.write('Epoch: {:02} | Time: {}m {}s'.format(epoch, epoch_mins, epoch_secs))
            log.write('\tTrain Loss: {:.3f} | Train PPL: {:7.3f}'.format(train_loss, math.exp(train_loss)))
            log.write('\tVal. Loss: {:.3f} | Val. PPL: {:7.3f}'.format(valid_loss, math.exp(valid_loss)))
            print('Epoch: {:02} | Time: {}m {}s'.format(epoch, epoch_mins, epoch_secs))
            print('\tTrain Loss: {:.3f} | Train PPL: {:7.3f}'.format(train_loss, math.exp(train_loss)))
            print('\tVal. Loss: {:.3f} | Val. PPL: {:7.3f}'.format(valid_loss, math.exp(valid_loss)))

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                self.saveModel(path_model)
            self.scheduler.step()

    def _transform(self, x):
        n_batch = 1 if len(x.shape) == 1 else x.shape[0]
        length = len(x) if len(x.shape) == 1 else x.shape[1]
        with torch.no_grad():
            tens = torch.tensor(x).view(length, n_batch).to(self.device)
            return torch.mean(self.model(tens, get_embeddings=True), axis=0).detach().cpu().numpy()

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

    def loadModel(self, path_model, device):
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
        self.k_size = parameters["k_size"]
        self.dropout = parameters["dropout"]
        self.max_length = parameters["max_length"]
        self.emb_dim = parameters["emb_dim"]
        self.ntokens = parameters["ntokens"]
        self.nclass = parameters["nclass"]
        self.nhead = parameters["nhead"]
        self.nlayers = parameters["nlayers"]
        self.tax_level = parameters["tax_level"]
        if device is not None:
            self.device = device
        else:
            self.device = parameters["device"] if self.device is None else self.device

        self.model = TransformerModel(self.ntokens, self.nclass, self.emb_dim, self.nhead,
                                      self.hid_dim, self.nlayers, self.dropout).to(self.device)
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
                      "nclass": self.nclass,
                      "max_length": self.max_length,
                      "emb_dim": self.emb_dim,
                      "ntokens": self.ntokens,
                      "nhead": self.nhead,
                      "nlayers": self.nlayers,
                      "k_size": self.k_size,
                      "tax_level": self.tax_level}
        torch.save(parameters, path_model.replace(".pt", "-parameters.pt"))

    def predict(self, x):
        self.model.eval()
        batch_size = x.shape[0] if len(x.shape) > 1 else 1
        x = torch.tensor(x).view(-1, batch_size)
        res = self.model(x)
        res = torch.nn.functional.softmax(res, dim=1)
        return res.detach().numpy()
        #return np.array(torch.argmax(res, axis=1).detach())

    def readPredictionWrapper(self):
        def readPrediction(L_read):
            res = self.preprocess_several_reads(list(L_read))
            return pd.Series([elem for elem in self.predict(res)])
        return F.pandas_udf(readPrediction, T.ArrayType(T.FloatType()))

    def read2genome(self, df):
        """
        transform kmers into a read embedding
        :param df: pyspark DataFrame
        :return: pyspark Dataframe with prediction
        """
        read_col = "read"
        prob_col = "prob"
        pred_col = "predict"
        df = df.withColumn(prob_col, self.readPredictionWrapper()(read_col)).toPandas()
        df[pred_col] = df[prob_col].apply(lambda x: np.argmax(x))
        df[prob_col] = df[prob_col].apply(lambda x: np.max(x))
        return df


def load_df_taxonomy_ref(path_tax_report):
    """
    load the dataframe associated with the taxonomy information
    :param path_tax_report: String, path to the csv file containing the taxonomy information
    :return:
    """
    df_taxonomy_ref = pd.read_csv(path_tax_report)
    df_taxonomy_ref = df_taxonomy_ref.applymap(str)
    return df_taxonomy_ref


if __name__ == "__main__":
    parser = parser_creator.ParserCreator()
    args = parser.parser_transformer_classifier()

    k = args.k_mer_size
    f_name = args.f_name
    if f_name[-3:] != ".pt":
        f_name += ".pt"
    print(f_name)

    n_steps = args.n_steps
    n_iterations = args.n_iterations
    batch_size = args.batch_size
    id_gpu = [int(x) for x in args.id_gpu.split(',')]
    n_cpus = args.n_cpus
    torch.set_num_threads(n_cpus)
    learning_rate = args.learning_rate
    emb_dim = args.embedding_size
    hid_dim = args.hidden_size
    dropout = args.dropout  # ref 0.1
    path_kmer2vec = args.path_kmer2vec
    nhead = args.nhead
    nlayers = args.nlayers
    tax_level = args.tax_level
    df_taxonomy_ref = load_df_taxonomy_ref(os.path.join(os.path.dirname(root_folder),
                                                                     "data/taxonomy/tax_report.csv"))
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
    model = TransformerClassifier(k, max_length=max_length, learning_rate=learning_rate, id_gpu=id_gpu, batch_size=batch_size,
                                  hid_dim=hid_dim,  dropout=dropout, n_iterations=n_iterations, emb_dim=emb_dim,
                                  nhead=nhead, nlayers=nlayers, tax_level=tax_level, f_name=f_name)
    print("Creating field iterator")
    model.create_field_iterator(path_data_train, path_data_valid, path_kmer2vec)
    print("Creating Model")
    model.create_model()
    hdfs.create_dir(os.path.join(path_analysis, "read2vec"), mode="local")
    clip = 1.
    print("Fit Model")
    model.fit(epochs=n_steps, clip=clip, path_model=os.path.join(path_analysis, "read2genome", f_name))
