from read2vec import Read2Vec

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import AdaptiveLogSoftmaxWithLoss

from torchtext.data import BPTTIterator, ReversibleField
from torchtext.datasets import LanguageModelingDataset
from torchtext import vocab

import numpy as np
import random
import math
import time
from tqdm import tqdm
import dill as pickle

import sys
import os

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

root_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, os.path.join(root_folder, "utils"))
import parser_creator
import hdfs_functions as hdfs
import logger


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.dropout = dropout
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional=True)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = [src sent len, batch size]
        embedded = self.dropout(self.embedding(src))
        # embedded = [src sent len, batch size, emb dim]
        # self.rnn.flatten_parameters()
        outputs, hidden = self.rnn(embedded)
        # outputs = [src sent len, batch size, hid dim * num directions]
        # hidden = [n layers * num directions, batch size, hid dim]

        # hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        # outputs are always from the last layer

        # hidden [-2, :, : ] is the last of the forwards RNN
        # hidden [-1, :, : ] is the last of the backwards RNN

        # initial decoder hidden is final hidden state of the forwards and backwards
        #  encoder RNNs fed through a linear layer
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))

        # outputs = [src sent len, batch size, enc hid dim * 2]
        # hidden = [batch size, dec hid dim]

        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super(Attention, self).__init__()

        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim

        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [src sent len, batch size, enc hid dim * 2]
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]

        # repeat encoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # hidden = [batch size, src sent len, dec hid dim]
        # encoder_outputs = [batch size, src sent len, enc hid dim * 2]

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        # energy = [batch size, src sent len, dec hid dim]

        attention = self.v(energy).squeeze(2)
        # attention= [batch size, src len]

        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention, cutoffs=None, div_value=None):
        super(Decoder, self).__init__()

        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.attention = attention
        self.cutoffs = cutoffs
        self.div_value = div_value

        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)

        self.dropout = nn.Dropout(dropout)

        if self.cutoffs is not None and self.div_value is not None:
            self.adaptiveSoftmax = AdaptiveLogSoftmaxWithLoss((enc_hid_dim * 2) + dec_hid_dim + emb_dim,
                                                              output_dim,
                                                              cutoffs=self.cutoffs,
                                                              div_value=self.div_value)
        else:
            self.out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
            self.adaptiveSoftmax = None

    def forward(self, input, hidden, encoder_outputs):

        # input = [batch size]
        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [src sent len, batch size, enc hid dim * 2]
        input = input.unsqueeze(0)
        # input = [1, batch size]

        embedded = self.dropout(self.embedding(input))
        # embedded = [1, batch size, emb dim]

        a = self.attention(hidden, encoder_outputs)
        # a = [batch size, src len]
        a = a.unsqueeze(1)
        # a = [batch size, 1, src len]

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # encoder_outputs = [batch size, src sent len, enc hid dim * 2]

        weighted = torch.bmm(a, encoder_outputs)

        # weighted = [batch size, 1, enc hid dim * 2]

        weighted = weighted.permute(1, 0, 2)

        # weighted = [1, batch size, enc hid dim * 2]

        rnn_input = torch.cat((embedded, weighted), dim=2)

        # rnn_input = [1, batch size, (enc hid dim * 2) + emb dim]
        # self.rnn.flatten_parameters()
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))

        # output = [sent len, batch size, dec hid dim * n directions]
        # hidden = [n layers * n directions, batch size, dec hid dim]

        # sent len, n layers and n directions will always be 1 in this decoder, therefore:
        # output = [1, batch size, dec hid dim]
        # hidden = [1, batch size, dec hid dim]
        # this also means that output == hidden
        assert (output == hidden).all()

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        if self.adaptiveSoftmax is None:
            output = self.out(torch.cat((output, weighted, embedded), dim=1))
            # output = [bsz, output dim]
        else:
            output = self.adaptiveSoftmax.log_prob(torch.cat((output, weighted, embedded), dim=1))
        return output, hidden.squeeze(0)


class Sequence2Sequence(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Sequence2Sequence, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5, get_embeddings=False):
        if get_embeddings:
            return self.encoder(src)

        # src = [src sent len, batch size]
        # trg = [trg sent len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time

        batch_size = src.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)

        # encoder_outputs is all hidden states of the input sequence, back and forwards
        # hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hidden = self.encoder(src)

        # first input to the decoder is the <sos> tokens
        output = trg[0, :]

        for t in range(1, max_len):
            output, hidden = self.decoder(output, hidden, encoder_outputs)
            top1 = output.max(1)[1]
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            output = (trg[t] if teacher_force else top1)
        return outputs


class Seq2Seq(Read2Vec):
    def __init__(self, k_size, max_length=30, learning_rate=0.001,
                 teacher_forcing_ratio=0.5, id_gpu=[-1], batch_size=64, hid_dim=256, drop_out=0.3,
                 n_iterations=None, p_permute=0.05, p_remove=0.05, emb_dim=None):
        """
        TODO
        :param embedding: 2-D Numpy array, matrix of embeddings
        :param dico_index: dictionary with the kmer and its index in the embeddings matrix
        :param reverse_index: dictionary with index as key and kmer as value
        :param k_size: int, kmer size
        :param step: int, The number of nucleotides that separate each k_mer
                        if step == k_size there there is only one cut
        """
        Read2Vec.__init__(self, None, None, None, k_size)
        self.max_length = max_length
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.hid_dim = hid_dim
        self.id_gpu = id_gpu
        self.device = torch.device("cuda:%s" % id_gpu[0] if id_gpu[0] != -1 else "cpu")
        # To be initialized with functions
        self.TEXT = None
        self.train_iterator = None
        self.valid_iterator = None
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.cutoffs = None
        self.div_value = 2
        self.drop_out = drop_out
        self.n_iterations = n_iterations
        self.p_permute = p_permute
        self.p_remove = p_remove
        self.MIN_LENGTH = 5
        self.emb_dim = emb_dim

    def create_field_iterator(self, path_data_train, path_data_valid, path_embeddings, nb_cutoffs=None):
        """
        Create the field and the iterator
        :param path_data_train: String, path of the training data
        :param path_data_valid: String, path of the validation data
        :param path_embeddings: String, path where is save the embeddings
        :return:
        """
        tokenize = lambda x: x.split()
        self.TEXT = ReversibleField(init_token='<sos>', eos_token='<eos>', tokenize=tokenize)
        lmDataset_train = LanguageModelingDataset(path_data_train, self.TEXT, newline_eos=True)
        lmDataset_valid = LanguageModelingDataset(path_data_valid, self.TEXT, newline_eos=True)
        if path_embeddings is not None:
            vectors = vocab.Vectors(os.path.join(path_embeddings, "embeddings.csv"), cache="./cache")
        else:
            vectors = None
        self.TEXT.build_vocab(lmDataset_train, lmDataset_valid, min_freq=2, vectors=vectors)
        stoi, itos, self.cutoffs = self.init_stoi_itos_cutoffs([path_data_train, path_data_valid],
                                                                   nb_cutoffs=nb_cutoffs)
        self.TEXT.vocab.itos = itos
        self.TEXT.vocab.stoi = stoi
        self.index_pad = self.TEXT.vocab.stoi['<pad>']
        self.index_unk = self.TEXT.vocab.stoi['<unk>']
        self.index_sos = self.TEXT.vocab.stoi['<sos>']
        self.index_eos = self.TEXT.vocab.stoi['<eos>']
        self.train_iterator = BPTTIterator(lmDataset_train, batch_size=self.batch_size, repeat=True,
                                           bptt_len=self.max_length, device=self.device, shuffle=True)
        self.valid_iterator = BPTTIterator(lmDataset_valid, batch_size=self.batch_size, repeat=True,
                                           bptt_len=self.max_length, device=self.device, shuffle=True)
        self.length_train = len(self.train_iterator)
        self.length_valid = len(self.valid_iterator)
        self.train_iterator = iter(self.train_iterator)
        self.valid_iterator = iter(self.valid_iterator)

    def create_model(self):
        """
        Initialize variables : model, criterion and optimizer
        :param hid_dim: Int, size of the hidden dimension
        :return:
        """

        def init_weights(m):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    nn.init.normal_(param.data, mean=0, std=0.01)
                else:
                    nn.init.constant_(param.data, 0)

        INPUT_DIM = self.TEXT.vocab.vectors.shape[0] if self.TEXT.vocab.vectors is not None else len(
            self.TEXT.vocab.stoi)
        OUTPUT_DIM = self.TEXT.vocab.vectors.shape[0] if self.TEXT.vocab.vectors is not None else len(
            self.TEXT.vocab.stoi)
        ENC_EMB_DIM = self.TEXT.vocab.vectors.shape[1] if self.TEXT.vocab.vectors is not None else self.emb_dim
        DEC_EMB_DIM = self.TEXT.vocab.vectors.shape[1] if self.TEXT.vocab.vectors is not None else self.emb_dim
        ENC_HID_DIM = self.hid_dim
        DEC_HID_DIM = self.hid_dim
        ENC_DROPOUT = self.drop_out
        DEC_DROPOUT = self.drop_out

        attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
        enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
        dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn, self.cutoffs,
                      self.div_value)

        self.model = Sequence2Sequence(enc, dec, self.device)

        if self.id_gpu[0] != -1 and len(self.id_gpu) > 1:
            # cuda0 = torch.device('cuda:%s' % self.id_gpu[0])
            # cuda1 = torch.device('cuda:%s' % self.id_gpu[1])
            # self.model.encoder.to(cuda0)
            # self.model.decoder.to(cuda1)
            self.model = nn.DataParallel(self.model, device_ids=self.id_gpu).to(self.device)
            # self.model = nn.DataParallel(self.model, device_ids=self.id_gpu)
            # self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=self.id_gpu).to(self.device)
            self.model.module.apply(init_weights)
            if self.TEXT.vocab.vectors is not None:
                self.model.module.encoder.embedding.weight.data.copy_(self.TEXT.vocab.vectors)
                self.model.module.decoder.embedding.weight.data.copy_(self.TEXT.vocab.vectors)
        else:
            self.model = self.model.to(self.device)
            self.model.apply(init_weights)
            if self.TEXT.vocab.vectors is not None:
                self.model.encoder.embedding.weight.data.copy_(self.TEXT.vocab.vectors)
                self.model.decoder.embedding.weight.data.copy_(self.TEXT.vocab.vectors)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.index_pad)

    def train(self, clip):

        self.model.train()

        epoch_loss = 0
        n_iterations = self.length_train if (
                    self.n_iterations is None or self.n_iterations > self.length_train) else self.n_iterations

        for i, batch in enumerate(tqdm(self.train_iterator, total=self.length_train)):
            if i > n_iterations:
                break
            src = self.batch_noiser(batch.text)
            trg = batch.text
            if src.shape[0] < self.MIN_LENGTH:
                continue
            self.optimizer.zero_grad()

            try:
                output = self.model(src, trg)
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory, retrying batch')
                    for p in self.model.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    output = self.model(src, trg)
                else:
                    raise e

            # trg = [trg sent len, batch size]
            # output = [trg sent len, batch size, output dim]

            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)

            # trg = [(trg sent len - 1) * batch size]
            # output = [(trg sent len - 1) * batch size, output dim]

            loss = self.criterion(output, trg)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)

            self.optimizer.step()

            epoch_loss += loss.detach().item()
        return epoch_loss * 1. / i

    def evaluate(self):
        self.model.eval()

        epoch_loss = 0
        n_iterations = self.length_valid if (
                    self.n_iterations is None or self.n_iterations > self.length_valid) else self.n_iterations

        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.valid_iterator, total=self.length_valid)):
                if i > n_iterations:
                    src = self.batch_noiser(batch.text)
                    embeddings, _ = self.model(src[:, :2], None, 0, get_embeddings=True)
                    break
                src = self.batch_noiser(batch.text)
                trg = batch.text
                if src.shape[0] < self.MIN_LENGTH:
                    continue
                try:
                    output = self.model(src, trg, 0)  # turn off teacher forcing
                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        print('| WARNING: ran out of memory, retrying batch')
                        for p in self.model.parameters():
                            if p.grad is not None:
                                del p.grad  # free some memory
                        torch.cuda.empty_cache()
                        output = self.model(src, trg, 0)  # turn off teacher forcing
                    else:
                        raise e
                # trg = [trg sent len, batch size]
                # output = [trg sent len, batch size, output dim]

                output = output[1:].view(-1, output.shape[-1])
                trg = trg[1:].view(-1)

                # trg = [(trg sent len - 1) * batch size]
                # output = [(trg sent len - 1) * batch size, output dim]

                loss = self.criterion(output, trg)

                epoch_loss += loss.detach().item()
        return epoch_loss * 1. / i

    def fit(self, n_epoch=10, clip=1, path_model="./seq2seq_attention.pt"):
        """
        Train the model with a certain number of epochs
        :param n_epoch: Int, number of training epoch
        :param clip: The clip option for the gradient descent
        :param path_model: String, The complete path where is saved the model
        :return:
        """
        print("Model :")
        print(self.model)

        def epoch_time(start_time, end_time):
            elapsed_time = end_time - start_time
            elapsed_mins = int(elapsed_time / 60)
            elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
            return elapsed_mins, elapsed_secs

        print("Number of batchs in the train iterator : %s" % self.length_train)
        print("Number of batchs in the valid iterator : %s" % self.length_valid)
        best_valid_loss = float('inf')
        for epoch in range(n_epoch):
            log.write("Epoch numero : %s" % epoch)
            if torch.cuda.is_available():
                log.write("Memory allocated %s" % torch.cuda.memory_allocated())
            start_time = time.time()

            train_loss = self.train(clip)
            torch.cuda.empty_cache()
            valid_loss = self.evaluate()

            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                self.saveModel(path_model)

            log.write('Time %sm %ss' % (epoch_mins, epoch_secs))
            if train_loss < 50 and valid_loss < 50:
                log.write('\tTrain Loss: {0:.3f} | Train PPL: {1:.3f}'.format(train_loss, math.exp(train_loss)))
                log.write('\t Val. Loss: {0:.3f} |  Val. PPL: {1:.3f}'.format(valid_loss, math.exp(valid_loss)))
            else:
                log.write('\tTrain Loss: {0:.3f}'.format(train_loss))
                log.write('\t Val. Loss: {0:.3f}'.format(valid_loss))
            torch.cuda.empty_cache()

    def sampleToTensor(self, X):
        """
        Transform a sample to a tensor
        :param X: numpy array, index ok kmers
        :return: Pytorch Tensor
        """
        res = torch.tensor(X).to(self.device)
        if len(res.shape) == 1:
            return res.view(len(res), 1)  # [sample length, batch size]
        return res.T

    def _transform(self, x):
        """
        transform kmers into a read embedding
        :param x: 1-D numpy array, index of the kmer in reverse_index (int to kmer)
        :return: 1-D Numpy array
        """
        with torch.no_grad():
            input_tensor = self.sampleToTensor(x)
            embeddings, _ = self.model(input_tensor, None, get_embeddings=True)
            return embeddings[-1].detach().cpu().numpy()[0]

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
                # Input to tensor
                input_tensor = self.sampleToTensor(x_batch)
                # Get embeddings
                embeddings, _ = self.model(input_tensor, None, get_embeddings=True)
                L_res.append(embeddings[-1].detach().cpu().numpy())
            return np.concatenate(L_res)

    def loadModel(self, path_model):
        """
        :param path_model: String, The complete path where is saved the model
        :return:
        """
        assert path_model[-3:] == ".pt", "The file name should end by .pt (.pytorch)"
        self.TEXT = pickle.load(open(path_model.replace(".pt", "-field.pt"), 'rb'))
        # Replace default index_pad and index_unk
        self.index_pad = self.TEXT.vocab.stoi['<pad>']
        self.index_unk = self.TEXT.vocab.stoi[' UNK ']
        parameters = pickle.load(open(path_model.replace(".pt", "-parameters.pt"), 'rb'))
        self.hid_dim = parameters["hid_dim"]
        self.drop_out = parameters["drop_out"]
        self.cutoffs = parameters["cutoffs"]
        self.div_value = parameters["div_value"]
        self.max_length = parameters["max_length"]
        self.emb_dim = parameters["emb_dim"]
        self.device = parameters["device"] if self.device is None else self.device
        # INPUT_DIM = self.TEXT.vocab.vectors.shape[0]
        # OUTPUT_DIM = self.TEXT.vocab.vectors.shape[0]
        # ENC_EMB_DIM = self.TEXT.vocab.vectors.shape[1]
        # DEC_EMB_DIM = self.TEXT.vocab.vectors.shape[1]
        INPUT_DIM = parameters["input_dim"]
        OUTPUT_DIM = parameters["input_dim"]
        ENC_EMB_DIM = parameters["emb_dim"]
        DEC_EMB_DIM = parameters["emb_dim"]
        ENC_HID_DIM = self.hid_dim
        DEC_HID_DIM = self.hid_dim
        ENC_DROPOUT = self.drop_out
        DEC_DROPOUT = self.drop_out

        attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
        enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
        dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn, self.cutoffs,
                      self.div_value)

        self.model = Sequence2Sequence(enc, dec, self.device).to(self.device)
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
        pickle.dump(self.TEXT, open(os.path.join(path_analysis, "read2vec", f_name.replace(".pt", "-field.pt")), 'wb'),
                    protocol=pickle.HIGHEST_PROTOCOL)
        parameters = {"device": self.device, "hid_dim": self.hid_dim, "drop_out": self.drop_out,
                      "cutoffs": self.cutoffs, "div_value": self.div_value, "max_length": self.max_length}
        if self.TEXT.vocab.vectors is not None:
            parameters["emb_dim"] = self.TEXT.vocab.vectors.shape[1]
            parameters["input_dim"] = self.TEXT.vocab.vectors.shape[0]
        else:
            parameters["emb_dim"] = self.emb_dim
            parameters["input_dim"] = len(self.TEXT.vocab.stoi)
        pickle.dump(parameters,
                    open(os.path.join(path_analysis, "read2vec", f_name.replace(".pt", "-parameters.pt")), 'wb'),
                    protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    ###################################
    # ------ Script's Parameters ------#
    ###################################
    parser = parser_creator.ParserCreator()
    args = parser.parser_seq2seq()

    k = args.k_mer_size
    window = args.window

    kmer_embeddings_algorithm = args.kmer_embeddings_algorithm
    parameter_learning = args.parameter_learning
    parameter_structu = "k_%s_w_%s" % (k, window)

    gene_catalog = args.gene_catalog
    dataset_name = args.dataset_name

    n_steps = args.n_steps
    n_iterations = args.n_iterations
    batch_size = args.batch_size
    id_gpu = [int(x) for x in args.id_gpu.split(',')]
    n_cpus = args.n_cpus
    torch.set_num_threads(n_cpus)
    learning_rate = args.learning_rate
    embedding_size = args.embedding_size
    hid_dim = args.embedding_size // 2
    nb_cutoffs = args.nb_cutoffs

    path_analysis = args.path_analysis
    path_data_train, path_data_valid = args.path_data.split(',')
    max_length = args.max_length

    path_log = args.path_log
    log_file = args.log_file

    log = logger.Logger(path_log, log_file, log_file,
                        variable_time={"k": k,  "embedding_size": embedding_size,
                                       "n_steps": n_steps, "batch_size": batch_size},
                        **vars(args))

    # create the model name
    f_name = "seq2seq_attention_noise"
    if kmer_embeddings_algorithm is not None:
        f_name = f_name + "_" + kmer_embeddings_algorithm
    f_name += "_k%s" % k
    if window is not None:
        f_name = f_name + "_w" + str(window)
    if parameter_learning is not None:
        f_name = f_name + "_%semb" % str(parameter_learning.split('_')[1])
    f_name = f_name + '_%semb' % str(embedding_size)
    f_name = f_name + '_%sit' % str(n_iterations)
    f_name = f_name + '_{:.0e}.pt'.format(learning_rate)

    assert f_name[-3:] == ".pt", "The file name should end by .pt (.pytorch)"

    path_embeddings = None
    if kmer_embeddings_algorithm is not None:
        log.write("loading embeedings")
        if gene_catalog is not None:
            path_embeddings = os.path.join(path_analysis, "kmer2vec", "genome", gene_catalog, kmer_embeddings_algorithm,
                                           parameter_structu, parameter_learning)
        else:
            path_embeddings = os.path.join(path_analysis, "kmer2vec", "metagenome", dataset_name,
                                           kmer_embeddings_algorithm, parameter_structu, parameter_learning)

    p_permute = 0.05
    p_remove = 0.02
    teacher_forcing_ratio = 0.5
    print("Initializing seq2seq model")
    model = Seq2Seq(k, max_length=max_length, teacher_forcing_ratio=teacher_forcing_ratio,
                    learning_rate=learning_rate, id_gpu=id_gpu, batch_size=batch_size, hid_dim=hid_dim,
                    n_iterations=n_iterations, p_permute=p_permute, p_remove=p_remove, emb_dim=embedding_size)
    print("Creating field iterator")
    model.create_field_iterator(path_data_train, path_data_valid, path_embeddings, nb_cutoffs)
    model.create_model()
    hdfs.create_dir(os.path.join(path_analysis, "read2vec"), mode="local")
    clip = 1
    model.fit(n_epoch=n_steps, clip=clip, path_model=os.path.join(path_analysis, "read2vec", f_name))


