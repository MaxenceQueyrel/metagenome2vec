import numpy as np
import pandas as pd

import os
import time
import math

import torch
from torch import nn
import torch.nn.functional as F
import random

from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score

import logging
from ray import tune
logger = logging.getLogger(tune.__name__)
logger.setLevel(level=logging.CRITICAL)

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

from metagenome2vec.utils.string_names import *
from metagenome2vec.data_processing.metagenomeNNDataset import item_batch_to_tensor

#############################
######## Class Model ########
#############################

class DeepSets(nn.Module):
    def __init__(self, phi, rho, mil_layer, device):
        super(DeepSets, self).__init__()
        self.phi = phi
        self.rho = rho
        self.mil_layer = mil_layer
        self.device = device
        if mil_layer == "attention":
            self.attention = nn.Sequential(
                nn.Linear(self.phi.last_hidden_size, self.phi.last_hidden_size // 3),
                nn.Tanh(),
                nn.Linear(self.phi.last_hidden_size // 3, 1)
            ).to(self.device)

    def forward(self, x):
        # compute the representation for each data point
        x = self.phi.forward(x)
        A = None
        # sum up the representations
        if self.mil_layer == "sum":
            x = torch.sum(x, dim=1, keepdim=True)
        if self.mil_layer == "max":
            x = torch.max(x, dim=1, keepdim=True)[0]
        if self.mil_layer == "attention":
            A = self.attention(x)
            A = F.softmax(A, dim=1)
            x = torch.bmm(torch.transpose(A, 2, 1), x)

        # compute the output
        out = self.rho.forward(x)
        return out, A

    def forward_batch(self, X):
        res = torch.stack([self(x.unsqueeze(0))[0] for x in X]).view(len(X), -1)
        try:
            return res.squeeze(1)
        except:
            return res

    def predict(self, X, threshold=0.5):
        self.eval()
        with torch.no_grad():
            if self.rho.output_size < 2:
                return torch.ge(torch.sigmoid(self.forward_batch(X)), threshold).int()
            return torch.argmax(torch.nn.functional.softmax(self.forward_batch(X), dim=1), dim=1)

    def predict_proba(self, X, threshold=0.5):
        self.eval()
        with torch.no_grad():
            if self.rho.output_size < 2:
                y_prob = torch.sigmoid(self.forward_batch(X)).float()
                return torch.ge(y_prob, threshold).int(), y_prob
            y_prob = torch.nn.functional.softmax(self.forward_batch(X), dim=1)
            return torch.argmax(y_prob, dim=1), y_prob

    def predict_with_attention(self, X, threshold=0.5):
        self.eval()
        with torch.no_grad():
            out, A = self.forward(X)
            if self.rho.output_size < 2:
                y_prob = torch.sigmoid(out)
                y = torch.ge(y_prob, threshold)
                return y, y_prob, A
            y_prob = torch.nn.functional.softmax(self.forward_batch(X), dim=1)
            y = torch.argmax(y_prob, dim=1)
            return y, y_prob, A

    def score(self, X, y_, threshold=0.5):
        self.eval()
        y = self.predict(X, threshold)
        score = (y.int() == y_.int()).sum().item()
        return score * 1. / len(X)


class Phi(nn.Module):
    def __init__(self, embed_size, hidden_init=200, n_layer=1, dropout=0.2):
        super(Phi, self).__init__()
        layer_size = [embed_size, hidden_init]
        n_layer -= 1
        for i in range(n_layer):
            hidden_init = hidden_init // 2
            layer_size.append(hidden_init)
        self.layers = []
        for i in range(len(layer_size) - 1):
            self.layers.append(nn.Linear(layer_size[i], layer_size[i + 1]))
            self.layers.append(nn.LeakyReLU())
            self.layers.append(nn.Dropout(dropout))
        self.nets = nn.Sequential(
            *self.layers[:-1]  # Remove the last drop out
        )
        self.last_hidden_size = layer_size[-1]

    def forward(self, x):
        return self.nets(x)


class Rho(nn.Module):
    def __init__(self, phi_hidden_size, hidden_init=100, n_layer=1, dropout=0.2, output_size=1):
        super(Rho, self).__init__()
        self.output_size = output_size
        layer_size = [phi_hidden_size, hidden_init]
        n_layer -= 1
        for i in range(n_layer):
            hidden_init = hidden_init // 2
            layer_size.append(hidden_init)
        self.layers = []
        for i in range(len(layer_size) - 1):
            self.layers.append(nn.Linear(layer_size[i], layer_size[i + 1]))
            self.layers.append(nn.LeakyReLU()),
            self.layers.append(nn.Dropout(dropout))
        self.layers.append(nn.Linear(layer_size[-1], output_size))
        self.nets = nn.Sequential(
            *self.layers
        )

    def forward(self, x):
        return self.nets(x)


#############################
##### Learning functions ####
#############################


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.kaiming_normal_(param.data, a=0.01)
        else:
            nn.init.constant_(param.data, 0)


def train(model, loader, optimizer, criterion, clip=-1):
    model.train()
    epoch_loss = 0
    for _, (X, y_, _) in enumerate(loader):
        # Initialize the optimizer
        optimizer.zero_grad()
        # Compute the result of data in the batch
        y = model.forward_batch(X)
        if model.rho.output_size == 1:
            loss = criterion(y, y_)  # Compute the loss value
        else:
            loss = criterion(y, y_.long())
        loss.backward()
        if clip >= 0.:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(loader)


def evaluate(model, loader, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for _, (X, y_, _) in enumerate(loader):
            y = model.forward_batch(X)
            if model.rho.output_size == 1:
                loss = criterion(y, y_)  # Compute the loss value
            else:
                loss = criterion(y, y_.long())
            epoch_loss += loss.item()
    return epoch_loss / len(loader)


def score(model, loader, average="macro", multi_class="raise"):
    model.eval()
    with torch.no_grad():
        y_, y_pred, y_prob = prediction(model, loader)
        acc = accuracy_score(y_, y_pred)
        f1 = f1_score(y_, y_pred, average=average)
        pre = precision_score(y_, y_pred, average=average)
        rec = recall_score(y_, y_pred, average=average)
        auc = roc_auc_score(y_, y_prob, multi_class=multi_class)
        return acc, pre, rec, f1, auc



def prediction(model, loader):
    model.eval()
    with torch.no_grad():
        L_y, L_y_pred, L_y_prob = [], [], []
        for i, (X, y_, _) in enumerate(loader):
            L_y.append(y_)
            y_pred, y_prob = model.predict_proba(X)
            L_y_pred.append(y_pred)
            L_y_prob.append(y_prob)
        y_ = torch.cat(L_y).cpu().detach().numpy()
        y_pred = torch.cat(L_y_pred).cpu().detach().numpy()
        y_prob = torch.cat(L_y_prob).cpu().detach().numpy()
    return y_, y_pred, y_prob


def fit(model, loader_train, loader_valid, optimizer, criterion, n_epoch, clip=-1, scheduler=None,
        early_stopping=5, path_model="./deepsets.pt", is_optimization=False):
    best_valid_loss = np.inf
    cpt_epoch_no_improvement = 0
    for epoch in range(n_epoch):
        start_time = time.time()

        train_loss = train(model, loader_train, optimizer, criterion, clip)
        valid_loss = evaluate(model, loader_valid, criterion)

        cpt_epoch_no_improvement += 1
        if scheduler is not None:
            scheduler.step()

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            cpt_epoch_no_improvement = 0
            best_valid_loss = valid_loss
            if not is_optimization:
                torch.save(model.state_dict(), path_model)

        if cpt_epoch_no_improvement == early_stopping:
            if not is_optimization:
                print("Stopping earlier because no improvement")
                model.load_state_dict(torch.load(path_model))
            return model

        if not is_optimization:
            print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
            try:
                print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
                print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
            except OverflowError:
                print(f'\tTrain Loss: {train_loss:.3f}')
                print(f'\t Val. Loss: {valid_loss:.3f}')
    return model


def table_prediction(model, dataset, device=torch.device("cpu")):
    if model.mil_layer == "attention":
        df_res = pd.DataFrame(columns=["subject.id", "group", "prediction", "percentage_prediction",
                                   "tax_1", "attention_1", "tax_2", "attention_2", "tax_3", "attention_3",
                                   "tax_4", "attention_4", "tax_5", "attention_5"])
        for i in range(len(dataset)):
            fasta_id = dataset.IDs[i]
            group = dataset.labels[i]
            x, y, g = item_batch_to_tensor(dataset[i], device)
            y, y_prob, A = model.predict_with_attention(x.unsqueeze(0))
            A = A.cpu().detach().numpy().squeeze()
            A_index = np.argsort(A)[::-1].copy()
            if A.shape == ():
                A = np.array([A])
            g_best = g[A_index]
            A_best = A[A_index]
            # fig = plt.figure()
            # ax = fig.add_axes([0,0,1,1])
            # ax.bar(g, A)
            d_tax_prob = {}

            for j in range(1, 6):
                if j == 1 or len(g_best) >= j:
                    d_tax_prob["%sa_tax" % j] = g_best[j-1].item()
                    d_tax_prob["%sb_prob" % j] = A_best[j-1].item()
                else:
                    d_tax_prob["%sa_tax" % j] = -1
                    d_tax_prob["%sb_prob" % j] = -1
            df_res.loc[i] = [fasta_id, group, y.int().item(), torch.max(y_prob).item()] + sorted(d_tax_prob.values())
        return df_res
    else:
        df_res = pd.DataFrame(columns=["subject.id", "group", "prediction", "percentage_prediction"])
        for i in range(len(dataset)):
            fasta_id = dataset.IDs[i]
            group = dataset.labels[i]
            x, y, g = item_batch_to_tensor(dataset[i], device)
            y, y_prob = model.predict_proba(x.unsqueeze(0))
            df_res.loc[i] = [fasta_id, group, y.int().item(), y_prob.item()]
        return df_res


def genome_dummies_and_normalize_count(df, col_features):
    count_sum_name = "count_sum"
    col_features.remove(count_name)
    genome_dummies = pd.get_dummies(df[genome_name], prefix=genome_name)
    df = pd.concat([df, genome_dummies], axis=1)
    df_genome_count = df[[id_subject_name, count_name]].groupby(id_subject_name).sum().reset_index()
    df_genome_count = df_genome_count.rename(columns={count_name: count_sum_name})
    df = pd.merge(df, df_genome_count, on=id_subject_name)
    df[col_features] = df[col_features].values / df[[count_name]].values
    df[count_name] = df[count_name] / df[count_sum_name]
    del df[count_sum_name]
    return df


# def train_evaluate(params, X, y_, embed_size, output_size, cv=10, test_size=0.2, device=torch.device("cpu")):
#     mean_accuracy, std_accuracy = cross_val_score_for_optimization(X, y_, params, embed_size, output_size, cv, test_size, device)
#     tune.report(mean_accuracy=mean_accuracy, std_accuracy=std_accuracy)


# def train_evaluate_factory(X, y_, parameterization, embed_size, output_size, cv=10, test_size=0.2, device=torch.device("cpu")):
#     def train_evaluate_decorator(function):
#         def wrapper(*args, **kwargs):
#             mean_accuracy, std_accuracy = function(X, y_, parameterization, embed_size, output_size, cv, test_size, device)
#             tune.report(mean_accuracy=mean_accuracy, std_accuracy=std_accuracy)
#         return wrapper
#     return train_evaluate_decorator

