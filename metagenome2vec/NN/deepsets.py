import numpy as np
import pandas as pd

import os
import time
import math
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
import random

from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

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
from metagenome2vec.data_processing.metagenomeNNDataset import MetagenomeNNDataset
from metagenome2vec.NN.utils import get_features

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


def item_batch_to_tensor(item, device):
    d = torch.tensor(item[0], dtype=torch.float, device=device)
    t = item[1]
    g = torch.tensor(item[2], dtype=torch.int, device=device)
    return d, t, g


def collate_fn(batch, device):
    data, target, genome = [], [], []
    for elem in batch:
        d, t, g = item_batch_to_tensor(elem, device)
        data.append(d)
        target.append(t)
        genome.append(g)
    return data, torch.tensor(target, dtype=torch.float, device=device), genome


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


def train_test_split_mil(X, y_, n_splits=1, test_size=0.2):
    """
    Initialise a train and a valid fold
    :param X: Matrix
    :param y_: class
    :param n_splits: int, number of split in the stratifiedKFolds
    :param test_size: float, percentage of the test size
    :return: X_train, y_train, X_valid, y_valid
    """
    X[group_name] = y_
    col_features = get_features(X)
    X_uniq = X.drop_duplicates(subset=[id_subject_name, group_name])[[id_subject_name, group_name]]
    sss = StratifiedShuffleSplit(n_splits, test_size=test_size)
    for id_train, id_valid in sss.split(X_uniq[id_subject_name], X_uniq[group_name]):
        id_train = X[id_subject_name].isin(X_uniq[id_subject_name].iloc[id_train])
        id_valid = X[id_subject_name].isin(X_uniq[id_subject_name].iloc[id_valid])
        X_train, X_valid, y_train, y_valid = X[id_train].reset_index(drop=True).drop(group_name, axis=1), X[id_valid].reset_index(drop=True).drop(group_name, axis=1), y_[id_train], y_[id_valid]
        scalar = StandardScaler()
        X_train[col_features] = scalar.fit_transform(X_train[col_features])
        X_valid[col_features] = scalar.transform(X_valid[col_features])
        yield X_train, X_valid, y_train, y_valid


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


def score(model, loader, average="macro", all_metrics=False, multi_class="raise"):
    model.eval()
    with torch.no_grad():
        y_, y_pred, y_prob = prediction(model, loader)
        acc = accuracy_score(y_, y_pred)
        if all_metrics:
            f1 = f1_score(y_, y_pred, average=average)
            pre = precision_score(y_, y_pred, average=average)
            rec = recall_score(y_, y_pred, average=average)
            auc = roc_auc_score(y_, y_prob, multi_class=multi_class)
            return acc, pre, rec, f1, auc
        return acc


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
        early_stopping=5, path_model="./", name_model=None):
    best_valid_loss = np.inf
    cpt_epoch_no_improvement = 0
    if name_model is None:
        name_model = "deepsets.pt"
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
            torch.save(model.state_dict(), os.path.join(path_model, name_model))

        if cpt_epoch_no_improvement == early_stopping:
            print("Stopping earlier because no improvement")
            model.load_state_dict(torch.load(os.path.join(path_model, name_model)))
            return model

        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        try:
            print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
        except OverflowError:
            print(f'\tTrain Loss: {train_loss:.3f}')
            print(f'\t Val. Loss: {valid_loss:.3f}')
    return model


def fit_for_optimization(model, loader_train, loader_valid, optimizer, criterion, n_epoch, clip,
                         scheduler=None, early_stopping=5):
    best_valid_loss = np.inf
    cpt_epoch_no_improvement = 0
    for _ in range(n_epoch):
        train(model, loader_train, optimizer, criterion, clip)
        valid_loss = evaluate(model, loader_valid, criterion)
        cpt_epoch_no_improvement += 1
        if valid_loss < best_valid_loss:
            cpt_epoch_no_improvement = 0
            best_valid_loss = valid_loss
        if cpt_epoch_no_improvement == early_stopping:
            return model
        if scheduler is not None:
            scheduler.step()
    return model


def cross_val_score_for_optimization(X, y_, params, embed_size, output_size, cv=10, test_size=0.2, device=torch.device("cpu")):
    scores = np.zeros(cv)
    for i, (X_train, X_valid, y_train, y_valid) in enumerate(tqdm(train_test_split_mil(X, y_, n_splits=cv, test_size=test_size), total=cv)):
        dataset_train = MetagenomeNNDataset(X_train, y_train)
        dataset_valid = MetagenomeNNDataset(X_valid, y_valid)
        params_loader = {'batch_size': params[batch_size],
                         'collate_fn': lambda x: collate_fn(x, device),
                         'shuffle': True}
        loader_train = DataLoader(dataset_train, **params_loader)
        loader_valid = DataLoader(dataset_valid, **params_loader)
        # init model
        phi = Phi(embed_size, params[hidden_init_phi], params[n_layer_phi], params[dropout])
        rho = Rho(phi.last_hidden_size, params[hidden_init_rho], params[n_layer_rho], params[dropout], output_size)
        model = DeepSets(phi, rho, params[mil_layer], device).to(device)
        #model.apply(init_weights)
        optimizer = optim.Adam(model.parameters(), lr=params[learning_rate], weight_decay=params[weight_decay])
        #optimizer = optim.SGD(model.parameters(), lr=params[learning_rate], weight_decay=params[weight_decay])
        if output_size <= 2:
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()
        # fitting and scoring
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.3)
        model = fit_for_optimization(model, loader_train, loader_valid, optimizer, criterion, params[n_epoch],
                                     params[clip], scheduler=scheduler)
        scores[i] = score(model, loader_valid)
    return np.mean(scores), np.std(scores)


def cross_val_score(X, y_, path_model, params, embed_size, output_size, cv=10, test_size=0.2, device=torch.device("cpu"), prediction_best_model_name=None):
    best_acc = -1
    A_acc_train = np.zeros(cv)
    A_pre_train = np.zeros(cv)
    A_rec_train = np.zeros(cv)
    A_f1_train = np.zeros(cv)
    A_auc_train = np.zeros(cv)
    A_acc_valid = np.zeros(cv)
    A_pre_valid = np.zeros(cv)
    A_rec_valid = np.zeros(cv)
    A_f1_valid = np.zeros(cv)
    A_auc_valid = np.zeros(cv)
    A_fit_time = np.zeros(cv)
    A_score_time = np.zeros(cv)
    for i, (X_train, X_valid, y_train, y_valid) in enumerate(tqdm(train_test_split_mil(X, y_, n_splits=cv, test_size=test_size), total=cv)):
        dataset_train = MetagenomeNNDataset(X_train, y_train)
        dataset_valid = MetagenomeNNDataset(X_valid, y_valid)
        params_loader = {'batch_size': params[batch_size],
                         'collate_fn': lambda x: collate_fn(x, device),
                         'shuffle': True}
        loader_train = DataLoader(dataset_train, **params_loader)
        loader_valid = DataLoader(dataset_valid, **params_loader)
        # model
        phi = Phi(embed_size, params[hidden_init_phi], params[n_layer_phi], params[dropout])
        rho = Rho(phi.last_hidden_size, params[hidden_init_rho], params[n_layer_rho], params[dropout], output_size)
        model = DeepSets(phi, rho, params[mil_layer], device).to(device)
        #model.apply(init_weights)
        optimizer = optim.Adam(model.parameters(), lr=params[learning_rate], weight_decay=params[weight_decay])
        #optimizer = optim.SGD(model.parameters(), lr=params[learning_rate], weight_decay=params[weight_decay])
        if output_size <= 2:
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.3)
        # fitting
        d = time.time()
        model = fit(model, loader_train, loader_valid, optimizer, criterion, params[n_epoch], params[clip], scheduler,
            path_model=path_model, name_model=".checkpoint.pt")
        A_fit_time[i] = time.time() - d
        # scoring on train
        d = time.time()
        acc, pre, rec, f1, auc = score(model, loader_train, all_metrics=True)
        A_acc_train[i] = acc
        A_pre_train[i] = pre
        A_rec_train[i] = rec
        A_f1_train[i] = f1
        A_auc_train[i] = auc
        # scoring on valid
        acc, pre, rec, f1, auc = score(model, loader_valid, all_metrics=True)
        A_score_time[i] = time.time() - d
        A_acc_valid[i] = acc
        A_pre_valid[i] = pre
        A_rec_valid[i] = rec
        A_f1_valid[i] = f1
        A_auc_valid[i] = auc
        if prediction_best_model_name is not None and best_acc < acc:
            best_acc = acc
            if params[mil_layer] == "attention":
                table_prediction_with_attention(model, dataset_train).to_csv(os.path.join(path_model, "train_" + prediction_best_model_name), index=False)
                table_prediction_with_attention(model, dataset_valid).to_csv(os.path.join(path_model, "valid_" + prediction_best_model_name), index=False)
            else:
                table_prediction(model, dataset_train).to_csv(os.path.join(path_model, "train_" + prediction_best_model_name), index=False)
                table_prediction(model, dataset_valid).to_csv(os.path.join(path_model, "valid_" + prediction_best_model_name), index=False)
    return {"test_accuracy": A_acc_valid,
            "fit_time": A_fit_time, "score_time": A_score_time,
            "test_accuracy": A_acc_valid, "train_accuracy": A_acc_train,
            "test_f1": A_f1_valid, "train_f1": A_f1_train,
            "test_precision": A_pre_valid, "train_precision": A_pre_train,
            "test_recall": A_rec_valid, "train_recall": A_rec_train,
            "test_roc_auc": A_auc_valid, "train_roc_auc": A_auc_train}


def table_prediction(model, dataset, device=torch.device("cpu")):
    df_res = pd.DataFrame(columns=["subject.id", "group", "prediction", "percentage_prediction"])
    for i in range(len(dataset)):
        fasta_id = dataset.IDs[i]
        group = dataset.labels[i]
        x, y, g = item_batch_to_tensor(dataset[i], device)
        y, y_prob = model.predict_proba(x.unsqueeze(0))
        df_res.loc[i] = [fasta_id, group, y.int().item(), y_prob.item()]
    return df_res


def table_prediction_with_attention(model, dataset, device=torch.device("cpu")):
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

