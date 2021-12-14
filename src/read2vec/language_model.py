

from .read2vec import Read2Vec

from io import open
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import os

import time
import math

import numpy as np


MAX_LENGHT = 30


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, device):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, self.hidden_size)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.device = torch.device(device)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)


class AttnDecoderRNN(nn.Module):
    def __init__(self, output_size, hidden_size, device, dropout_p=0.1, max_length=MAX_LENGHT+1):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.device = torch.device(device)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)


class LanguageModel(Read2Vec):
    def __init__(self, embeddings, dico_index, k_size, path_model, device="cuda", path_data_for_learning=None, n_iters=1e5):
        """
        :param read2vec: Read2Vec object
        :param embeddings: 2-D Numpy array, matrix of embeddings
        :param dico_index: dictionary with the kmer and its index in the embeddings matrix
        :param k_size: int, kmer size
        :param path_model: String, Path where is save the trained model
        :param path_data_for_learning: String, path of the data to train
        :param n_iters
        """
        Read2Vec.__init__(self, embeddings, dico_index, k_size)
        self.EOS_token = 1
        self.SOS_token = 0
        self.path_model = path_model
        self.path_data_for_learning = path_data_for_learning
        self.device = torch.device(device)
        # Add SOS EOS and absente kmer in dico index and embeddings
        tokens = np.zeros((2, self.embeddings.shape[1]))
        self.embeddings = np.concatenate((tokens, self.embeddings))
        self.dico_index = {str(k): v + 2 for k, v in self.dico_index.items()}

        dico_index["SOS"] = self.SOS_token
        dico_index["EOS"] = self.EOS_token
        if not os.path.isfile(path_model):
            print("Init encoder")
            self.encoder = EncoderRNN(self.embeddings.shape[0], self.embeddings.shape[1], self.device)#.to(self.device)
            print("Chargement des embeddings encoder")
            self.encoder.embedding.weight.data.copy_(torch.from_numpy(self.embeddings))
            print("Parallelization encoder")
            self.encoder = nn.DataParallel(self.encoder)
            self.encoder.to(device)
            print("Init decoder")
            attn_decoder = AttnDecoderRNN(self.embeddings.shape[0], self.embeddings.shape[1], self.device)#.to(self.device)
            print("Chargement des embeddings decoder")
            attn_decoder.embedding.weight.data.copy_(torch.from_numpy(self.embeddings))
            print("Parallelization decoder")
            attn_decoder = nn.DataParallel(attn_decoder)
            attn_decoder.to(device)
            print("Debut tu training")
            self.trainIters(self.encoder, attn_decoder, n_iters, n_iters/20)
            torch.save(self.encoder, path_model)
        else:
            self.encoder = torch.load(path_model)

    def prepare_data(self, path_data, length_sequence):
        f_read = open(path_data, "r")
        pairs = []
        for line in f_read:
            line = line.split()
            i = 0
            while i * length_sequence < len(line):
                pairs.append([' '.join(line[i * length_sequence:i * length_sequence + length_sequence]),
                              ' '.join(line[i * length_sequence:i * length_sequence + length_sequence])])
                i += 1
        f_read.close()
        return pairs

    def indexesFromSentence(self, sentence):
        res = []
        for word in sentence.split(' '):
            try:
                res.append(self.dico_index[word])
            except:
                continue
        return res

    def tensorFromSentence(self, sentence):
        indexes = self.indexesFromSentence(sentence)
        indexes.append(self.EOS_token)
        return torch.tensor(indexes, dtype=torch.long, device=self.device).view(-1, 1)

    def tensorsFromPair(self, pair):
        tensor = self.tensorFromSentence(pair[0])
        return tensor, tensor

    def train(self, input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
              max_length=MAX_LENGHT+1, teacher_forcing_ratio=0.5):
        encoder_hidden = encoder.initHidden()
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=self.device)

        loss = 0

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(
                input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = torch.tensor([[self.SOS_token]], device=self.device)

        decoder_hidden = encoder_hidden

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                loss += criterion(decoder_output, target_tensor[di])
                decoder_input = target_tensor[di]  # Teacher forcing

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input

                loss += criterion(decoder_output, target_tensor[di])
                if decoder_input.item() == self.EOS_token:
                    break

        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        return loss.item() / target_length

    def asMinutes(self, s):
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    def timeSince(self, since, percent):
        now = time.time()
        s = now - since
        es = s / (percent)
        rs = es - s
        return '%s (- %s)' % (self.asMinutes(s), self.asMinutes(rs))

    def trainIters(self, encoder, decoder, n_iters, print_every=1000, learning_rate=0.01):
        start = time.time()
        print_loss_total = 0  # Reset every print_every
        plot_loss_total = 0  # Reset every plot_every

        encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
        decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
        pairs = self.prepare_data(self.path_data_for_learning, MAX_LENGHT)
        training_pairs = [self.tensorsFromPair(random.choice(pairs))
                          for i in range(n_iters)]
        criterion = nn.NLLLoss()

        for iter in range(1, n_iters + 1):
            print(iter)
            training_pair = training_pairs[iter - 1]
            input_tensor = training_pair[0]
            target_tensor = training_pair[1]

            loss = self.train(input_tensor, target_tensor, encoder,
                         decoder, encoder_optimizer, decoder_optimizer, criterion)
            print_loss_total += loss
            plot_loss_total += loss

            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (self.timeSince(start, iter / n_iters),
                                             iter, iter / n_iters * 100, print_loss_avg))

    def getEmbedding(self, encoder, sentence):
        with torch.no_grad():
            input_tensor = self.tensorFromSentence(sentence)
            input_length = input_tensor.size()[0]
            encoder_hidden = encoder.initHidden()

            for ei in range(input_length):
                encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                         encoder_hidden)
            encoder_hidden = encoder_hidden.cpu().numpy()
            return encoder_hidden.reshape(encoder_hidden.shape[2])

    def compute(self, tokens):
        """
        Transform a read into embedding
        :param tokens: list, list of kmer

        :return: embeddings, numpy array
        """
        return self.getEmbedding(self.encoder, " ".join(tokens))

    def predict(self, L_read):
        """
        transform all reads in L_read into embeddings
        :param L_read: List of String
        :return: 2D Numpy array
        """
        L_read = self.preprocess_several_reads(L_read)
        L_res = []
        for read in L_read:
            res = self.compute(read)
            if res is not None:  # No error
                L_res.append(res)
        del L_read
        return np.array(L_res)
