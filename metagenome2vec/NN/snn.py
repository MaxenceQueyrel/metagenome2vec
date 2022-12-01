import numpy as np

import time
import math

import torch
from torch import nn

import metagenome2vec.NN.utils as utils
from metagenome2vec.utils.string_names import *


#############################
######## Class Model ########
#############################

class SiameseNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=50, n_layer_after_flatten=1, n_layer_before_flatten=1, device="cpu",
                 activation_function="nn.ReLU"):
        super(SiameseNetwork, self).__init__()
        assert n_layer_after_flatten >= 0 and n_layer_before_flatten >= 0, "Number of layers should be a positive integer"
        # Get the number of instance and the input dim
        self.n_instance, self.n_dim = input_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.n_layer_after_flatten = n_layer_after_flatten
        self.n_layer_before_flatten = n_layer_before_flatten
        self.activation_function = activation_function
        self.embed_dim = None
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")

        # to save the encoder dimension
        L_hidden_dim = []
        # initialize the encoder
        self.encoder = []
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
        # Create the last layers of the encoder
        self.encoder.append(nn.Linear(hidden_current, hidden_next))
        self.embed_dim = hidden_next
        # create sequentials
        self.encoder = nn.Sequential(
            *self.encoder
        )
        self.predictor = nn.Linear(hidden_next, 1)

    def get_coefs(self, coef, imputation=0.75, stage="encoder"):
        if stage == "encoder":
            return 1. + 1. * coef, coef * imputation
        return 1. + 3. * coef, coef * imputation

    def processing(self, *args):
        res = tuple([x.float().to(self.device) for x in args])
        if len(res) == 1:
            return res[0]
        return res

    def forward_once(self, x):
        # Forward pass
        output = self.encoder(x)
        return output

    def forward(self, input1, input2):
        # forward pass of input 1
        output1 = self.forward_once(input1)
        # forward pass of input 2
        output2 = self.forward_once(input2)
        dis = torch.abs(output1 - output2)
        output = self.predictor(dis)
        return output.flatten()

    def transform(self, sample):
        return self.forward_once(sample)

    def create_matrix_embeddings(self, dataloader, add_abundance="no"):
        """
        add_abundance: str, "no": without, "yes" with, "only" don't use embeddings
        """
        X_res = y_res = None
        for _, (d1, d2, y, y1) in enumerate(dataloader):
            # Get abundance from original data
            if add_abundance != "no":
                abundance = d1[:, :, 0].cpu().detach().numpy()
            # SNN embeddings
            if not add_abundance == "only":
                d1, d2, y = self.processing(d1, d2, y)
                X_tmp = self.transform(d1).cpu().detach().numpy()
            else:
                X_tmp = abundance
            # Concatenante embeddings and abundance to get final representation
            if add_abundance == "yes":
                X_tmp = np.concatenate((abundance, X_tmp), axis=1)
            y_tmp = y1.cpu().detach().numpy()
            X_res = X_tmp if X_res is None else np.concatenate((X_res, X_tmp))
            y_res = y_tmp if y_res is None else np.concatenate((y_res, y_tmp))
        return X_res, y_res


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on:
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, x0, x1, y):
        # euclidian distance
        diff = x0 - x1
        dist_sq = torch.sum(torch.pow(diff, 2), 1)
        dist = torch.sqrt(dist_sq)

        mdist = self.margin - dist
        dist = torch.clamp(mdist, min=0.0)
        loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
        loss = torch.sum(loss) / 2.0 / x0.size()[0]
        return loss


def train(model, loader, optimizer, clip=-1):
    model.train()
    train_loss = 0
    for _, data in enumerate(loader):
        d1, d2, y, _ = data
        d1, d2, y = model.processing(d1, d2, y)
        optimizer.zero_grad()
        output = model(d1, d2)
        loss = model.criterion(output, y)
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
        for _, data in enumerate(loader):
            d1, d2, y, _ = data
            d1, d2, y = model.processing(d1, d2, y)
            output = model(d1, d2)
            test_loss += model.criterion(output, y).item()
    test_loss /= len(loader.dataset)
    return test_loss


def fit(model, loader_train, loader_valid, optimizer, n_epoch, clip=-1, scheduler=None,
        early_stopping=5, path_model="./snn.pt", is_optimization=False):
    best_valid_loss = np.inf
    cpt_epoch_no_improvement = 0
    for epoch in range(n_epoch):
        start_time = time.time()
        train_loss = train(model, loader_train, optimizer, clip)
        valid_loss = evaluate(model, loader_valid)
        cpt_epoch_no_improvement += 1
        if scheduler is not None:
            scheduler.step()

        end_time = time.time()
        epoch_mins, epoch_secs = utils.epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            cpt_epoch_no_improvement = 0
            best_valid_loss = valid_loss
            if not is_optimization:
                torch.save(model.state_dict(), path_model)

        if cpt_epoch_no_improvement == early_stopping:
            print("Stopping earlier because no improvement")
            if not is_optimization:
                model.load_state_dict(torch.load(path_model))
            return

        if not is_optimization:
            print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
            try:
                print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
                print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
            except OverflowError:
                print(f'\tTrain Loss: {train_loss:.3f}')
                print(f'\t Val. Loss: {valid_loss:.3f}')



