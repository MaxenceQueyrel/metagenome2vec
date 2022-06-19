import numpy as np
import pandas as pd

import os
import time
import math
import sys
from tqdm import tqdm
import pickle
import abc

import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
import random

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, MaxAbsScaler

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

from metagenome2vec.utils import parser_creator
from metagenome2vec.utils import data_manager
from metagenome2vec.utils import file_manager
from metagenome2vec.utils.string_names import *

file_name_parameters = "hyper_parameters.pkl"
vae_name = 'vae.pt'
vae_fine_tuned_name = 'vae_fine_tuned.pt'

############################################
#### Functions to load and generate data ###
############################################

class metagenomeDataset(Dataset):
    def __init__(self, X, y_):
        self.data = X.copy()
        self.data[group_name] = y_.copy()
        self.IDs = pd.unique(self.data[id_subject_name])
        self.labels = self.data.drop_duplicates(subset=[id_subject_name])[group_name].to_numpy()
        del self.data[group_name]
        self.data = np.array([np.array(self.data[self.data[id_subject_name] == id_])[:, 1:] for id_ in self.IDs],
                             dtype=float)

    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


#############################
######## Class Model ########
#############################

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=50, n_layer_after_flatten=1, n_layer_before_flatten=1, device="cpu",
                 activation_function="nn.ReLU"):
        super(AutoEncoder, self).__init__()
        assert n_layer_after_flatten >= 0 and n_layer_before_flatten >= 0, "Number of layers should be a positive integer"
        # Get the number of instance and the input dim
        self.n_instance, self.n_dim = input_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.n_layer_after_flatten = n_layer_after_flatten
        self.n_layer_before_flatten = n_layer_before_flatten
        self.activation_function = activation_function
        self.reconstruction_function = nn.MSELoss(reduction="sum")
        self.embed_dim = None

    def get_coefs(self, coef, imputation=0.75, stage="encoder"):
        if stage == "encoder":
            return 1. + 1. * coef, coef * imputation
        return 1. + 3. * coef, coef * imputation

    def decode(self, z):
        return self.decoder(z)

    def processing(self, x):
        return x.float().to(self.device)

    @abc.abstractmethod
    def forward(self, x):
        raise NotImplementedError("Not implemented")

    @abc.abstractmethod
    def encode(self, x):
        raise NotImplementedError("Not implemented")

    @abc.abstractmethod
    def forward(self, x):
        raise NotImplementedError("Not implemented")

    @abc.abstractmethod
    def transform(self, sample):
        raise NotImplementedError("Not implemented")

    def create_matrix_embeddings(self, dataloader, add_abundance="no"):
        """
        add_abundance: str, "no": without, "yes" with, "only" don't use embeddings
        """
        X_res = y_res = None
        for batch_id, (data, y) in enumerate(dataloader):
            # Get abundance from original data
            if add_abundance != "no":
                abundance = data[:, :, 0].cpu().detach().numpy()
            # VAE embeddings
            if not add_abundance == "only":
                X_tmp = self.transform(self.processing(data)).cpu().detach().numpy()
            else:
                X_tmp = abundance
            # Concatenante embeddings and abundance to get final representation
            if add_abundance == "yes":
                X_tmp = np.concatenate((abundance, X_tmp), axis=1)
            y_tmp = y.cpu().detach().numpy()
            X_res = X_tmp if X_res is None else np.concatenate((X_res, X_tmp))
            y_res = y_tmp if y_res is None else np.concatenate((y_res, y_tmp))
        return X_res, y_res


class VAE(AutoEncoder):

    def __init__(self, input_dim, hidden_dim=50, n_layer_after_flatten=1, n_layer_before_flatten=1, device="cpu",
                 activation_function="nn.ReLU"):
        super().__init__(input_dim, hidden_dim, n_layer_after_flatten, n_layer_before_flatten, device,
                         activation_function)
        # to save the encoder dimension
        L_hidden_dim = []
        # initialize the encoder and the decoder
        self.encoder = []
        self.decoder = []
        # Initialize the current hidden and the next hidden
        hidden_current = self.n_dim
        hidden_next = self.hidden_dim
        # create the encoder linear before the flatten operation
        coef = 1.
        for i in range(self.n_layer_before_flatten):
            self.encoder.append(nn.Linear(hidden_current, hidden_next))
            L_hidden_dim.append((hidden_next, hidden_current))
            self.encoder.append(eval(self.activation_function)())
            div, coef = self.get_coefs(coef, stage="encoder")
            hidden_current, hidden_next = hidden_next, int(hidden_next / div)
        # Add the flatten layer and compute the new dimensions
        hidden_at_flatten = hidden_current
        self.encoder.append(nn.Flatten())
        hidden_current = self.n_instance * hidden_current
        div, coef = self.get_coefs(1., stage="decoder")
        hidden_next = int(hidden_current / div)
        # Add the encoder layers
        for i in range(self.n_layer_after_flatten):
            self.encoder.append(nn.Linear(hidden_current, hidden_next))
            L_hidden_dim.append((hidden_next, hidden_current))
            self.encoder.append(eval(self.activation_function)())
            div, coef = self.get_coefs(coef, stage="decoder")
            hidden_current, hidden_next = hidden_next, int(hidden_next / div)
        # Create the last layers of the ecnoder
        self.fc1 = nn.Linear(hidden_current, hidden_next)
        self.fc2 = nn.Linear(hidden_current, hidden_next)
        # definng the abstract variable
        self.embed_dim = hidden_next
        self.decoder.append(nn.Linear(hidden_next, hidden_current))
        # if self.n_layer_before_flatten == 0:
        #     self.decoder.append(nn.Tanh())
        # else:
        #     self.decoder.append(eval(self.activation_function)())
        if self.n_layer_before_flatten != 0:
            self.decoder.append(eval(self.activation_function)())
        # Add decoder layers
        for i in range(self.n_layer_after_flatten):
            self.decoder.append(nn.Linear(*L_hidden_dim.pop()))
            self.decoder.append(eval(self.activation_function)())
        # unflatten and compute new dimensions
        self.decoder.append(nn.Unflatten(1, torch.Size([self.n_instance, hidden_at_flatten])))
        for i in range(self.n_layer_before_flatten):
            self.decoder.append(nn.Linear(*L_hidden_dim.pop()))
            # if i != self.n_layer_before_flatten - 1:
            #     self.decoder.append(eval(self.activation_function)())
            # else:
            #     self.decoder.append(nn.Tanh())
            if i != self.n_layer_before_flatten - 1:
                self.decoder.append(eval(self.activation_function)())
        # create sequentials
        self.encoder = nn.Sequential(
            *self.encoder
        )
        self.decoder = nn.Sequential(
            *self.decoder  # [:-1]  # Remove the last drop out
        )

    def encode(self, x):
        if len(self.encoder) != 0:
            x = self.encoder(x)
        return self.fc1(x), self.fc2(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        # print("encode", z.min().min(), z.max().max())
        # print("decode", self.decode(z).min().min(), self.decode(z).max().max())
        return self.decode(z), mu, logvar

    def transform(self, sample):
        return self.reparameterize(*self.encode(sample))

    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self, x, recon_x, mu, logvar):
        # BCE = F.binary_cross_entropy(recon_x, x.view(-1, input_dim), reduction='sum')
        BCE = self.reconstruction_function(recon_x, x)
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD


class AE(AutoEncoder):

    def __init__(self, input_dim, hidden_dim=50, n_layer_after_flatten=1, n_layer_before_flatten=1, device="cpu",
                 activation_function="nn.ReLU"):
        super().__init__(input_dim, hidden_dim, n_layer_after_flatten, n_layer_before_flatten, device,
                         activation_function)
        # to save the encoder dimension
        L_hidden_dim = []
        # initialize the encoder and the decoder
        self.encoder = []
        self.decoder = []
        # Initialize the current hidden and the next hidden
        hidden_current = self.n_dim
        hidden_next = hidden_dim
        # create the encoder linear before the flatten operation
        coef = 1.
        for i in range(self.n_layer_before_flatten):
            self.encoder.append(nn.Linear(hidden_current, hidden_next))
            L_hidden_dim.append((hidden_next, hidden_current))
            self.encoder.append(eval(self.activation_function)())
            div, coef = self.get_coefs(coef)
            hidden_current, hidden_next = hidden_next, int(hidden_next / div)
        # Add the flatten layer and compute the new dimensions
        hidden_at_flatten = hidden_current
        self.encoder.append(nn.Flatten())
        hidden_current = self.n_instance * hidden_current
        div, coef = self.get_coefs(1.)
        hidden_next = int(hidden_current / div)
        # Add the encoder layers
        for i in range(self.n_layer_after_flatten):
            self.encoder.append(nn.Linear(hidden_current, hidden_next))
            L_hidden_dim.append((hidden_next, hidden_current))
            self.encoder.append(eval(self.activation_function)())
            div, coef = self.get_coefs(coef)
            hidden_current, hidden_next = hidden_next, int(hidden_next / div)
        # definng the abstract variable
        self.embed_dim = hidden_current
        # self.decoder.append(nn.Linear(hidden_next, hidden_current))
        # if self.n_layer_before_flatten == 0:
        #     self.decoder.append(nn.Tanh())
        # else:
        #     self.decoder.append(eval(self.activation_function)())
        if self.n_layer_before_flatten != 0:
            self.decoder.append(eval(self.activation_function)())
        # Add decoder layers
        for i in range(self.n_layer_after_flatten):
            self.decoder.append(nn.Linear(*L_hidden_dim.pop()))
            self.decoder.append(eval(self.activation_function)())
        # unflatten and compute new dimensions
        self.decoder.append(nn.Unflatten(1, torch.Size([self.n_instance, hidden_at_flatten])))
        for i in range(self.n_layer_before_flatten):
            self.decoder.append(nn.Linear(*L_hidden_dim.pop()))
            # if i != self.n_layer_before_flatten - 1:
            #     self.decoder.append(eval(self.activation_function)())
            # else:
            #     self.decoder.append(nn.Tanh())
            if i != self.n_layer_before_flatten - 1:
                self.decoder.append(eval(self.activation_function)())

        # create sequentials
        self.encoder = nn.Sequential(
            *self.encoder
        )
        self.decoder = nn.Sequential(
            *self.decoder  # [:-1]  # Remove the last drop out
        )

    def encode(self, x):
        return self.encoder(x)

    def transform(self, sample):
        return self.encode(sample)

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

    def loss_function(self, x, recon_x):
        return self.reconstruction_function(recon_x, x)


class FineTuner(nn.Module):
    def __init__(self, auto_encoder, n_output):
        super(FineTuner, self).__init__()
        self.auto_encoder = auto_encoder
        self.n_output = n_output
        if self.n_output > 1:
            self.reconstruction_function_finetune = nn.CrossEntropyLoss()
        else:
            self.reconstruction_function_finetune = nn.BCEWithLogitsLoss()
        self.predictor = nn.Linear(self.auto_encoder.embed_dim, self.n_output)

    def forward(self, x):
        x = self.auto_encoder.transform(x)
        return self.predictor(x)

    def loss_function_finetune(self, y, y_):
        return self.reconstruction_function_finetune(y, y_)

    def processing(self, y):
        return self.auto_encoder.processing(y).reshape(len(y), 1)

    def create_prediction(self, dataloader, threshold=0.5):
        y_pred = y_true = y_prob = None
        for batch_id, (data, y_) in enumerate(dataloader):
            data = self.auto_encoder.processing(data)
            y = self(data)
            y_p = y.cpu().detach().numpy()
            y_prob = y_p if y_prob is None else np.concatenate((y_prob, y_p))
            if self.n_output == 1:
                y = torch.ge(torch.sigmoid(y), threshold).int().cpu().detach().numpy().flatten()
            else:
                y = torch.argmax(torch.nn.functional.softmax(y, dim=1), dim=1)
            y_pred = y if y_pred is None else np.concatenate((y_pred, y))
            y_true = y_ if y_true is None else np.concatenate((y_true, y_))
        return y_pred, y_true, y_prob

    def score(self, dataloader, threshold=0.5):
        y_pred, y_true, _ = self.create_prediction(dataloader, threshold)
        return (y_pred == y_true).sum() / len(y_true)

#############################
##### Learning functions ####
#############################


def train_test_split(X, y_, col_features, n_splits=1, test_size=0.2):
    """
    Initialise a train and a valid fold
    :param X: Matrix
    :param y_: class
    :param col_features: List of str, feature columns used for normalization
    :param n_splits: int, number of split in the stratifiedKFolds
    :param test_size: float, percentage of the test size
    :return: X_train, y_train, X_valid, y_valid
    """
    X = X.copy()
    X[group_name] = y_.copy()
    X_uniq = X.drop_duplicates(subset=[id_subject_name, group_name])[[id_subject_name, group_name]]
    sss = StratifiedShuffleSplit(n_splits, test_size=test_size)
    for id_train, id_valid in sss.split(X_uniq[id_subject_name], X_uniq[group_name]):
        id_train = X[id_subject_name].isin(X_uniq[id_subject_name].iloc[id_train])
        id_valid = X[id_subject_name].isin(X_uniq[id_subject_name].iloc[id_valid])
        X_train, X_valid, y_train, y_valid = X[id_train].reset_index(drop=True).drop(group_name, axis=1),\
                                             X[id_valid].reset_index(drop=True).drop(group_name, axis=1), \
                                             y_[id_train].values, y_[id_valid].values
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


reconstruction_function = nn.MSELoss(reduction="sum")

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function_vae(recon_x, x, mu, logvar):
    #BCE = F.binary_cross_entropy(recon_x, x.view(-1, input_dim), reduction='sum')
    BCE = reconstruction_function(recon_x, x)
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


def loss_function_ae(recon_x, x):
    return reconstruction_function(recon_x, x)


def train_finetune(model, loader, optimizer, clip=-1):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(loader):
        if len(data) == 2:
            data, y_ = data
        else:
            data, _, y_, _ = data
        data = model.auto_encoder.processing(data)
        optimizer.zero_grad()
        y = model(data).to(model.auto_encoder.device)
        loss = model.loss_function_finetune(y, model.processing(y_))
        loss.backward()
        if clip >= 0.:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        train_loss += loss.item()
        optimizer.step()
    return train_loss / len(loader.dataset)


def evaluate_finetune(model, loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data in enumerate(loader):
            if len(data) == 2:
                data, y_ = data
            else:
                data, _, y_, _ = data
            data = model.auto_encoder.processing(data)
            y = model(data).to(model.auto_encoder.device)
            test_loss += model.loss_function_finetune(y, model.processing(y_)).item()

    test_loss /= len(loader.dataset)
    return test_loss


def train(model, loader, optimizer, clip=-1):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(loader):
        data = model.processing(data)
        optimizer.zero_grad()
        output = model(data)
        if isinstance(output, tuple):
            loss = model.loss_function(data, *output)
        else:
            loss = model.loss_function(data, output)
        loss.backward()
        if clip >= 0.:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        train_loss += loss.item()
        optimizer.step()
    return train_loss / len(loader.dataset)


def evaluate(model, loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(loader):
            data = model.processing(data)
            output = model(data)
            if isinstance(output, tuple):
                test_loss += model.loss_function(data, *output).item()
            else:
                test_loss += model.loss_function(data, output).item()
    test_loss /= len(loader.dataset)
    return test_loss


def fit(model, loader_train, loader_valid, optimizer, n_epoch, clip=-1, scheduler=None,
        early_stopping=5, path_model="./", name_model=None):
    best_valid_loss = np.inf
    cpt_epoch_no_improvement = 0
    is_fine_tune = model.__class__.__name__ == "FineTuner"
    if name_model is None:
        name_model = vae_name if not is_fine_tune else vae_fine_tuned_name
    for epoch in range(n_epoch):
        start_time = time.time()

        if is_fine_tune:
            train_loss = train_finetune(model, loader_train, optimizer, clip)
            valid_loss = evaluate_finetune(model, loader_valid)
        else:
            train_loss = train(model, loader_train, optimizer, clip)
            valid_loss = evaluate(model, loader_valid)
        cpt_epoch_no_improvement += 1
        if scheduler is not None:
            scheduler.step()

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            cpt_epoch_no_improvement = 0
            best_valid_loss = valid_loss
            if is_fine_tune:
                torch.save(model.auto_encoder.state_dict(), os.path.join(path_model, name_model))
            else:
                torch.save(model.state_dict(), os.path.join(path_model, name_model))

        if cpt_epoch_no_improvement == early_stopping:
            print("Stopping earlier because no improvement")
            if is_fine_tune:
                model.auto_encoder.load_state_dict(torch.load(os.path.join(path_model, name_model)))
            else:
                model.load_state_dict(torch.load(os.path.join(path_model, name_model)))
            return

        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        try:
            print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
        except OverflowError:
            print(f'\tTrain Loss: {train_loss:.3f}')
            print(f'\t Val. Loss: {valid_loss:.3f}')


def fit_for_optimization(model, loader_train, loader_valid, optimizer, n_epoch, clip,
                         scheduler=None, early_stopping=5):
    best_valid_loss = np.inf
    cpt_epoch_no_improvement = 0
    for epoch in range(n_epoch):
        train(model, loader_train, optimizer, clip)
        valid_loss = evaluate(model, loader_valid)
        cpt_epoch_no_improvement += 1
        if valid_loss < best_valid_loss:
            cpt_epoch_no_improvement = 0
            best_valid_loss = valid_loss
        if cpt_epoch_no_improvement == early_stopping:
            return
        if scheduler is not None:
            scheduler.step()


def cross_val_score_for_optimization(params, cv=10):
    scores = np.zeros(cv)
    for i, (X_train, X_valid, y_train, y_valid) in enumerate(tqdm(train_test_split(X, y_, col_features, n_splits=cv,
                                                                                       test_size=test_size), total=cv)):
        dataset_train = metagenomeDataset(X_train, y_train)
        dataset_valid = metagenomeDataset(X_valid, y_valid)
        input_dim = dataset_train.data.shape[1:]
        params_loader = {'batch_size': params[batch_size],
                         'shuffle': True}
        loader_train = DataLoader(dataset_train, **params_loader)
        loader_valid = DataLoader(dataset_valid, **params_loader)
        # init model
        model = NN(input_dim, hidden_dim=params[hidden_dim],
                   n_layer_before_flatten=params[n_layer_before_flatten],
                   n_layer_after_flatten=params[n_layer_after_flatten],
                   device=device, activation_function=params[activation_function]).to(device)
        #model.apply(init_weights)
        optimizer = optim.Adam(model.parameters(), lr=params[learning_rate], weight_decay=params[weight_decay])
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.3)
        # fitting and scoring
        fit_for_optimization(model, loader_train, loader_valid, optimizer, params[n_epoch], params[clip],
                             scheduler=scheduler)
        scores[i] = evaluate(model, loader_valid)
    return np.mean(scores), np.std(scores)


def cross_val_score(params, cv=10):
    A_score_train = np.zeros(cv)
    A_score_valid = np.zeros(cv)
    A_fit_time = np.zeros(cv)
    A_score_time = np.zeros(cv)
    best_score = np.inf
    for i, (X_train, X_valid, y_train, y_valid) in enumerate(tqdm(train_test_split(X, y_, col_features, n_splits=cv,
                                                                                       test_size=test_size), total=cv)):
        dataset_train = metagenomeDataset(X_train, y_train)
        dataset_valid = metagenomeDataset(X_valid, y_valid)
        params_loader = {'batch_size': params[batch_size],
                         'shuffle': True}
        loader_train = DataLoader(dataset_train, **params_loader)
        loader_valid = DataLoader(dataset_valid, **params_loader)
        # model
        model = NN(input_dim=params[input_dim], hidden_dim=params[hidden_dim],
                   n_layer_before_flatten=params[n_layer_before_flatten],
                   n_layer_after_flatten=params[n_layer_after_flatten],
                   device=device, activation_function=params[activation_function]).to(device)
        #vae.apply(init_weights)
        optimizer = optim.Adam(model.parameters(), lr=params[learning_rate], weight_decay=params[weight_decay])
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.3)
        # fitting
        d = time.time()
        fit(model, loader_train, loader_valid, optimizer, params[n_epoch], params[clip], scheduler,
            path_model=path_model, name_model=".checkpoint.pt")
        A_fit_time[i] = time.time() - d
        # scoring on train
        d = time.time()
        score_train = evaluate(model, loader_train)
        A_score_train[i] = score_train
        # scoring on valid
        score_valid = evaluate(model, loader_train)
        A_score_valid[i] = score_valid
        A_score_time[i] = time.time() - d
        if score_valid < best_score:
            best_score = score_valid
            torch.save(model.state_dict(), os.path.join(path_model, vae_name))
    return {"score_train": A_score_train,
            "score_valid": A_score_valid,
            "fit_time": A_fit_time, "score_time": A_score_time}


def train_evaluate(parameterization):
    mean_score, std_score = cross_val_score_for_optimization(parameterization, cv=cv)
    tune.report(mean_score=mean_score, std_score=std_score)


def mil_data_preprocessing(X, y_, genomes=None):
    X[group_name] = y_
    if genomes is None:  # create a set of all genomes (training mode)
        genomes = set(X[genome_name].drop_duplicates().tolist())
    else:  # keep only the ones in genomes (inference mode)
        X = X[X[genome_name].isin(genomes)]
    # create a set of all genomes by id
    X_gb = X[[id_subject_name, group_name, genome_name]].groupby([id_subject_name, group_name]).agg(
        {genome_name: lambda x: list(x)}).reset_index()
    # Keep only the genomes not in genomes to initialize their embeddings to 0
    X_gb[genome_name] = X_gb[genome_name].apply(lambda x: genomes.difference(x))
    X_gb = X_gb[X_gb[genome_name] != set()]
    X_gb = X_gb.explode(genome_name).reset_index(drop=True)
    L_feature = get_features(X)
    n_feature = len(L_feature)
    X_gb = pd.concat([X_gb, pd.DataFrame(np.zeros((X_gb.shape[0], n_feature)),
                                         columns=[count_name] + [str(x) for x in range(n_feature - 1)])], axis=1)
    # Reform the original dataset and ordre by id and genome
    X = X[[id_subject_name, group_name, genome_name] + L_feature]
    X = pd.concat([X, X_gb], axis=0).reset_index(drop=True)
    X_metadata = X[[id_subject_name, genome_name, group_name]]
    del X[genome_name]
    del X[group_name]
    X = pd.concat([X_metadata, X.groupby(id_subject_name).apply(lambda x: x / x[count_name].sum())], axis=1)
    X = X.sort_values([id_subject_name, genome_name])
    y_ = X[group_name].copy()
    del X[group_name]
    del X[genome_name]
    return X, y_, genomes


def get_features(X):
    return [count_name] + [col for col in X.columns.values if str(col).isdigit()]


def set_device(id_gpu):
    id_gpu = [int(x) for x in id_gpu.split(',')]
    if id_gpu != [-1]:
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(x) for x in id_gpu])
    return torch.device("cuda:%s" % id_gpu[0] if id_gpu != [-1] else "cpu")


def save_model(path_model, parameters, genomes):
    with open(os.path.join(path_model, file_name_parameters), 'wb') as fp:
        pickle.dump(parameters, fp)
    with open(os.path.join(path_model, 'genomes'), 'wb') as fp:
        pickle.dump(genomes, fp)


def load_model(path_model):
    with open(os.path.join(path_model, file_name_parameters), 'rb') as fp:
        parameters = pickle.load(fp)
    with open(os.path.join(path_model, 'genomes'), 'rb') as fp:
        genomes = pickle.load(fp)
    return parameters, genomes


if __name__ == "__main__":
    parser = parser_creator.ParserCreator()
    args = parser.parser_vae()
    # Script variables
    path_data = args.path_data
    path_metadata = args.path_metadata
    disease = args.disease
    dataset_name = args.dataset_name
    path_model = os.path.join(args.path_model, dataset_name)
    batch_size_ = args.batch_size
    n_steps_ = args.n_steps
    learning_rate_ = args.learning_rate
    weight_decay_ = args.weight_decay
    dropout_ = args.dropout
    clip_ = args.clip
    activation_function_= args.activation_function
    path_tmp = args.path_tmp_folder
    n_memory = args.n_memory * 1000 * 1024 * 1024  # To convert in giga
    computation_type = args.computation_type
    NN = VAE if computation_type == "vae" else AE

    if path_tmp is None:
        path_tmp = os.environ["TMP"] if "TMP" in os.environ else "~/"

    hidden_dim_, n_layer_before_flatten_, n_layer_after_flatten_ = [int(x) for x in args.vae_struct.split(",")]
    device = set_device(args.id_gpu)

    resources = {str(x.split(':')[0]): float(x.split(':')[1]) for x in args.resources.split(",")}
    D_resource = {"worker": 1, "cpu": 1, "gpu": 0}
    for resource in args.resources.split(","):
        name, value = resource.split(":")
        if name == "worker":
            D_resource[name] = int(value)
        else:
            D_resource[name] = float(value)

    tuning = args.tuning
    num_samples = args.n_iterations

    cv = args.cross_validation
    resources_per_trial = {"cpu": D_resource["cpu"], "gpu": D_resource["gpu"]}
    test_size = args.test_size

    # Load data
    X, y_ = data_manager.load_several_matrix_for_learning(path_data, path_metadata, disease)
    col_features = get_features(X)

    X, y_, genomes = mil_data_preprocessing(X, y_)
    input_dim_ = (len(genomes), X.shape[1] - 1)  # input dimension of the auto encoder
    path_file_parameters = os.path.join(path_model, file_name_parameters)

    params = {input_dim: input_dim_,
              batch_size: batch_size_,
              activation_function: activation_function_,
              n_epoch: n_steps_,
              learning_rate: learning_rate_,
              weight_decay: weight_decay_,
              hidden_dim: hidden_dim_,
              n_layer_before_flatten: n_layer_before_flatten_,
              n_layer_after_flatten: n_layer_after_flatten_,
              dropout: dropout_,
              clip: clip_}

    n_output = len(set(y_.tolist()))
    n_output = 1 if n_output == 2 else n_output

    file_manager.create_dir(path_model, mode="local")

    # Tune
    if tuning:
        os.environ["RAY_PICKLE_VERBOSE_DEBUG"] = "1"
        ray.init(num_cpus=np.int(np.ceil(D_resource["cpu"] * D_resource["worker"])),
                 object_store_memory=n_memory,
                 num_gpus=np.int(np.ceil(D_resource["gpu"] * D_resource["worker"])))
        parameters = [{"name": learning_rate, "type": "range", "bounds": [1e-4, 1e-1], "log_scale": True},
                      {"name": weight_decay, "type": "choice", "values": [0.0, 1e-3], "value_type": "float"},
                      {"name": dropout, "type": "choice", "values": [0.0, 0.1]},
                      {"name": activation_function, "type": "choice", "values": ["nn.ReLU", "nn.LeakyReLU"], "value_type": "str"},
                      {"name": batch_size, "type": "fixed", "value": 6, "value_type": "int"},
                      {"name": n_epoch, "type": "fixed", "value": 100, "value_type": "int"},
                      {"name": n_layer_before_flatten, "type": "choice", "values": [2, 3, 4], "value_type": "int"},
                      {"name": n_layer_after_flatten, "type": "choice", "values": [2, 3, 4], "value_type": "int"},
                      {"name": hidden_dim, "type": "choice", "values": [30, 50], "value_type": "int"},
                      {"name": clip, "type": "fixed", "value": -1., "value_type": "float"}]

        ax = AxClient(enforce_sequential_optimization=False, random_seed=SEED)
        metric_to_tune = "mean_score"
        ax.create_experiment(
            name="autoencoder_experiment",
            parameters=parameters,
            objective_name=metric_to_tune,
        )

        algo = AxSearch(ax_client=ax, max_concurrent=D_resource["worker"])
        scheduler = AsyncHyperBandScheduler()

        analyse = tune.run(train_evaluate,
                           num_samples=num_samples,
                           search_alg=algo,
                           scheduler=scheduler,
                           mode="min",
                           metric=metric_to_tune,
                           verbose=1,
                           resources_per_trial=resources_per_trial,
                           local_dir=os.path.join(path_tmp, "ray_results_" + dataset_name))

        analyse.dataframe().to_csv(os.path.join(path_model, "tuning_results.csv"), index=False)
        best_parameters = analyse.get_best_config(metric=metric_to_tune, mode="min")
        best_parameters[input_dim] = input_dim_
        save_model(path_model, best_parameters, genomes)

    # Train and test with cross validation
    if os.path.exists(path_file_parameters):
        print("Best parameters used")
        with open(path_file_parameters, 'rb') as fp:
            best_parameters = pickle.load(fp)
    else:
        print("Default parameters used")
        best_parameters = {}

    # Change the parameters with the best ones
    for k, v in best_parameters.items():
        params[k] = v

    # cross val scores
    print(params)
    scores = cross_val_score(params, cv)
    print(scores)
