import numpy as np
import pandas as pd

import os
import time
import math
import sys
from tqdm import tqdm
import json

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader
import random

from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from ax.service.ax_client import AxClient
import logging
import ray
from ray import tune
from ray.tune.suggest.ax import AxSearch
from ray.tune.schedulers import AsyncHyperBandScheduler
logger = logging.getLogger(tune.__name__)
logger.setLevel(level=logging.CRITICAL)

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

root_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, os.path.join(root_folder, "utils"))

import parser_creator
import data_manager
import hdfs_functions as hdfs
from string_names import *

############################################
############ Global variables ##############
############################################

learning_rate = "learning_rate"
batch_size = "batch_size"
n_epoch = "n_epoch"
mil_layer = "mil_layer"
weight_decay = "weight_decay"
step_size = "step_size"
gamma = "gamma"
hidden_init_phi = "hidden_init_phi"
hidden_init_rho = "hidden_init_rho"
n_layer_phi = "n_layer_phi"
n_layer_rho = "n_layer_rho"
dropout = "dropout"
clip = "clip"


############################################
#### Functions to load and generate data ###
############################################


class metagenomeDataset(Dataset):
    def __init__(self, X, y_):
        self.data = X
        self.data[group_name] = y_
        self.labels = self.data.drop_duplicates(subset=[id_subject_name])[group_name].to_numpy()
        self.IDs = pd.unique(self.data[id_subject_name])
        self.D_data = {
            id_subject: self.data[self.data[id_subject_name] == id_subject].drop([group_name, genome_name, id_subject_name],
                                                                               axis=1).to_numpy() for id_subject in self.IDs}
        self.D_genome = {id_subject: self.data[self.data[id_subject_name] == id_subject][genome_name].to_numpy() for id_subject in
                         self.IDs}

    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, idx):
        id_subject, labels = self.IDs[idx], self.labels[idx]
        if isinstance(idx, slice):
            items = [self.D_data[x] for x in id_subject]
            genomes = [self.D_genome[x] for x in id_subject]
        else:
            items = self.D_data[id_subject]
            genomes = self.D_genome[id_subject]
        return items, labels, genomes


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


def train_test_split_mil(X, y_, col_features, n_splits=1, test_size=0.2):
    """
    Initialise a train and a valid fold
    :param X: Matrix
    :param y_: class
    :param col_features: List of str, feature columns used for normalization
    :param n_splits: int, number of split in the stratifiedKFolds
    :param test_size: float, percentage of the test size
    :return: X_train, y_train, X_valid, y_valid
    """
    X[group_name] = y_
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
    for it, (X, y_, _) in enumerate(loader):
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
        for it, (X, y_, _) in enumerate(loader):
            y = model.forward_batch(X)
            if model.rho.output_size == 1:
                loss = criterion(y, y_)  # Compute the loss value
            else:
                loss = criterion(y, y_.long())
            epoch_loss += loss.item()
    return epoch_loss / len(loader)


def score(model, loader, all_metrics=False):
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
    for epoch in range(n_epoch):
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


def cross_val_score_for_optimization(params, cv=10):
    scores = np.zeros(cv)
    for i, (X_train, X_valid, y_train, y_valid) in enumerate(tqdm(train_test_split_mil(X, y_, col_features, n_splits=cv,
                                                                                       test_size=test_size), total=cv)):
        dataset_train = metagenomeDataset(X_train, y_train)
        dataset_valid = metagenomeDataset(X_valid, y_valid)
        params_loader = {'batch_size': params[batch_size],
                         'collate_fn': lambda x: collate_fn(x, device),
                         'shuffle': True}
        loader_train = DataLoader(dataset_train, **params_loader)
        loader_valid = DataLoader(dataset_valid, **params_loader)
        # init model
        phi = Phi(embed_size, params[hidden_init_phi], params[n_layer_phi], params[dropout])
        rho = Rho(phi.last_hidden_size, params[hidden_init_rho], params[n_layer_rho], params[dropout], output_size)
        deepsets = DeepSets(phi, rho, params[mil_layer], device).to(device)
        #deepsets.apply(init_weights)
        optimizer = optim.Adam(deepsets.parameters(), lr=params[learning_rate], weight_decay=params[weight_decay])
        #optimizer = optim.SGD(deepsets.parameters(), lr=params[learning_rate], weight_decay=params[weight_decay])
        if output_size <= 2:
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()
        # fitting and scoring
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.3)
        deepsets = fit_for_optimization(deepsets, loader_train, loader_valid, optimizer, criterion, params[n_epoch],
                                        params[clip], scheduler=scheduler)
        scores[i] = score(deepsets, loader_valid)
    return np.mean(scores), np.std(scores)


def cross_val_score(params, cv=10, prediction_best_model_name=None):
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
    for i, (X_train, X_valid, y_train, y_valid) in enumerate(tqdm(train_test_split_mil(X, y_, col_features, n_splits=cv,
                                                                                       test_size=test_size), total=cv)):
        dataset_train = metagenomeDataset(X_train, y_train)
        dataset_valid = metagenomeDataset(X_valid, y_valid)
        params_loader = {'batch_size': params[batch_size],
                         'collate_fn': lambda x: collate_fn(x, device),
                         'shuffle': True}
        loader_train = DataLoader(dataset_train, **params_loader)
        loader_valid = DataLoader(dataset_valid, **params_loader)
        # model
        phi = Phi(embed_size, params[hidden_init_phi], params[n_layer_phi], params[dropout])
        rho = Rho(phi.last_hidden_size, params[hidden_init_rho], params[n_layer_rho], params[dropout], output_size)
        deepsets = DeepSets(phi, rho, params[mil_layer], device).to(device)
        #deepsets.apply(init_weights)
        optimizer = optim.Adam(deepsets.parameters(), lr=params[learning_rate], weight_decay=params[weight_decay])
        #optimizer = optim.SGD(deepsets.parameters(), lr=params[learning_rate], weight_decay=params[weight_decay])
        if output_size <= 2:
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.3)
        # fitting
        d = time.time()
        deepsets = fit(deepsets, loader_train, loader_valid, optimizer, criterion, params[n_epoch], params[clip], scheduler,
            path_model=path_model, name_model=".checkpoint.pt")
        A_fit_time[i] = time.time() - d
        # scoring on train
        d = time.time()
        acc, pre, rec, f1, auc = score(deepsets, loader_train, all_metrics=True)
        A_acc_train[i] = acc
        A_pre_train[i] = pre
        A_rec_train[i] = rec
        A_f1_train[i] = f1
        A_auc_train[i] = auc
        # scoring on valid
        acc, pre, rec, f1, auc = score(deepsets, loader_valid, all_metrics=True)
        A_score_time[i] = time.time() - d
        A_acc_valid[i] = acc
        A_pre_valid[i] = pre
        A_rec_valid[i] = rec
        A_f1_valid[i] = f1
        A_auc_valid[i] = auc
        if prediction_best_model_name is not None and best_acc < acc:
            best_acc = acc
            if params[mil_layer] == "attention":
                table_prediction_with_attention(deepsets, dataset_train).to_csv(os.path.join(path_model, "train_" + prediction_best_model_name), index=False)
                table_prediction_with_attention(deepsets, dataset_valid).to_csv(os.path.join(path_model, "valid_" + prediction_best_model_name), index=False)
            else:
                table_prediction(deepsets, dataset_train).to_csv(os.path.join(path_model, "train_" + prediction_best_model_name), index=False)
                table_prediction(deepsets, dataset_valid).to_csv(os.path.join(path_model, "valid_" + prediction_best_model_name), index=False)
    return {"test_accuracy": A_acc_valid,
            "fit_time": A_fit_time, "score_time": A_score_time,
            "test_accuracy": A_acc_valid, "train_accuracy": A_acc_train,
            "test_f1": A_f1_valid, "train_f1": A_f1_train,
            "test_precision": A_pre_valid, "train_precision": A_pre_train,
            "test_recall": A_rec_valid, "train_recall": A_rec_train,
            "test_roc_auc": A_auc_valid, "train_roc_auc": A_auc_train}


def table_prediction(model, dataset):
    df_res = pd.DataFrame(columns=["subject.id", "group", "prediction", "percentage_prediction"])
    for i in range(len(dataset)):
        fasta_id = dataset.IDs[i]
        group = dataset.labels[i]
        x, y, g = item_batch_to_tensor(dataset[i], device)
        y, y_prob = model.predict_proba(x.unsqueeze(0))
        df_res.loc[i] = [fasta_id, group, y.int().item(), y_prob.item()]
    return df_res


def table_prediction_with_attention(model, dataset):
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
        df_res.loc[i] = [fasta_id, group, y.int().item(), torch.max(y_prob).item()] + [v for k, v in sorted(d_tax_prob.items())]
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


def train_evaluate(parameterization):
    mean_accuracy, std_accuracy = cross_val_score_for_optimization(parameterization, cv=cv)
    tune.report(mean_accuracy=mean_accuracy, std_accuracy=std_accuracy)


def set_device(id_gpu):
    id_gpu = [int(x) for x in id_gpu.split(',')]
    if id_gpu != [-1]:
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(x) for x in id_gpu])
    return torch.device("cuda:%s" % id_gpu[0] if id_gpu != [-1] else "cpu")


def get_features(X):
    return [count_name] + [col for col in X.columns.values if str(col).isdigit()]


if __name__ == "__main__":
    parser = parser_creator.ParserCreator()
    args = parser.parser_deepsets()

    # Script variables
    path_data = args.path_data
    path_metadata = args.path_metadata
    disease = args.disease
    path_save = args.path_save
    dataset_name = args.dataset_name
    path_model = args.path_model
    batch_size_ = args.batch_size
    n_steps_ = args.n_steps
    learning_rate_ = args.learning_rate
    weight_decay_ = args.weight_decay
    dropout_ = args.dropout
    clip_ = args.clip
    path_tmp = args.path_tmp_folder
    if path_tmp is None:
        path_tmp = os.environ["TMP"] if "TMP" in os.environ else "~/"

    hidden_init_phi_, hidden_init_rho_, n_layer_phi_, n_layer_rho_ = [int(x) for x in args.deepsets_struct.split(",")]

    resources = {str(x.split(':')[0]): float(x.split(':')[1]) for x in args.resources.split(",")}
    D_resource = {"worker": 1, "cpu": 1, "gpu": 0}
    for resource in args.resources.split(","):
        name, value = resource.split(":")
        if name == "worker":
            D_resource[name] = int(value)
        else:
            D_resource[name] = float(value)

    n_memory = args.n_memory * 1000 * 1024 * 1024  # To convert in giga
    tuning = args.tuning
    num_samples = args.n_iterations
    device = set_device(args.id_gpu)

    cv = args.cross_validation
    resources_per_trial = {"cpu": D_resource["cpu"], "gpu": D_resource["gpu"]}
    test_size = args.test_size

    params = {batch_size: batch_size_,
              n_epoch: n_steps_,
              learning_rate: learning_rate_,
              mil_layer: "attention",
              weight_decay: weight_decay_,
              hidden_init_phi: hidden_init_phi_,
              hidden_init_rho: hidden_init_rho_,
              n_layer_phi: n_layer_phi_,
              n_layer_rho: n_layer_rho_,
              dropout: dropout_,
              clip: clip_}

    # Load data
    X, y_ = data_manager.load_several_matrix_for_learning(path_data, path_metadata, disease)
    output_size = 1 if len(np.unique(y_)) == 2 else len(np.unique(y_))
    average = "binary" if output_size <= 2 else "micro"  # when compute scores, change average if binary or multi class
    multi_class = "raise" if output_size <= 2 else "ovr"
    col_features = get_features(X)
    embed_size = X.shape[1] - 2  # - id_subject and genome

    file_best_parameters = os.path.join(path_model, 'best_parameters')
    # Tune
    if tuning:
        ray.init(num_cpus=np.int(np.ceil(D_resource["cpu"] * D_resource["worker"])),
                 #memory=n_memory,
                 num_gpus=np.int(np.ceil(D_resource["gpu"] * D_resource["worker"])))
        parameters = [{"name": learning_rate, "type": "range", "bounds": [1e-5, 1e-2], "log_scale": True},
                      {"name": weight_decay, "type": "fixed", "value": 0.0002},
                      {"name": dropout, "type": "choice", "values": [0.0, 0.3]},
                      {"name": batch_size, "type": "fixed", "value": 6},
                      {"name": n_epoch, "type": "fixed", "value": 100},
                      {"name": n_layer_phi, "type": "choice", "values": [1, 3]},
                      {"name": n_layer_rho, "type": "choice", "values": [1, 3]},
                      {"name": hidden_init_phi, "type": "choice", "values": [50, 100]},
                      {"name": hidden_init_rho, "type": "choice", "values": [50, 100]},
                      {"name": mil_layer, "type": "choice", "values": ["sum", "attention"]},
                      {"name": clip, "type": "fixed", "value": -1.}]

        ax = AxClient(enforce_sequential_optimization=False, random_seed=SEED)
        metric_to_tune = "mean_accuracy"
        ax.create_experiment(
            name="deepsets_experiment",
            parameters=parameters,
            objective_name=metric_to_tune,
        )

        algo = AxSearch(ax_client=ax, max_concurrent=D_resource["worker"])
        scheduler = AsyncHyperBandScheduler()

        analyse = tune.run(train_evaluate,
                           num_samples=num_samples,
                           search_alg=algo,
                           scheduler=scheduler,
                           mode="max",
                           metric=metric_to_tune,
                           verbose=1,
                           resources_per_trial=resources_per_trial,
                           local_dir=os.path.join(path_tmp, "ray_results_" + dataset_name)
                           )
        hdfs.create_dir(path_model, mode="local")
        analyse.dataframe().to_csv(os.path.join(path_model, "tuning_results.csv"), index=False)
        # best_parameters, values = ax.get_best_parameters()
        best_parameters = analyse.get_best_config(metric=metric_to_tune, mode="max")
        with open(file_best_parameters, 'w') as fp:
            json.dump(best_parameters, fp)

    # Train and test with cross validation
    if os.path.exists(file_best_parameters):
        print("Best parameters used")
        with open(file_best_parameters, 'r') as fp:
            best_parameters = json.load(fp)
    else:
        print("Default parameters used")
        best_parameters = {}

    # Change the parameters with the best ones
    for k, v in best_parameters.items():
        params[k] = v
    print(params)
    # cross val scores
    scores = cross_val_score(params, cv, prediction_best_model_name=dataset_name + ".csv")
    data_manager.write_file_res_benchmarck_classif(path_save, dataset_name, "deepsets", scores)
