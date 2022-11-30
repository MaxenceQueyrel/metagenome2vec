import numpy as np
import time
import math
import abc

import torch
from torch import nn

import metagenome2vec.NN.utils as utils
from metagenome2vec.utils.string_names import *


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
    for _, data in enumerate(loader):
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
        for _, data in enumerate(loader):
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
    for _, (data, _, _) in enumerate(loader):
        data = torch.concat([x.unsqueeze(0) for x in data])
        optimizer.zero_grad()
        output = model(data)
        loss = model.loss_function(data, *output) if isinstance(output, tuple) else  model.loss_function(data, output)
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
        for _, (data, _, _) in enumerate(loader):
            output = model(data)
            test_loss += model.loss_function(data, *output).item() if isinstance(output, tuple) else model.loss_function(data, output).item()
    test_loss /= len(loader.dataset)
    return test_loss


def fit(model, loader_train, loader_valid, optimizer, n_epoch, clip=-1, scheduler=None,
        early_stopping=5, path_model="./vae.pt", is_optimization=False):
    best_valid_loss = np.inf
    cpt_epoch_no_improvement = 0
    is_fine_tune = model.__class__.__name__ == "FineTuner"
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
        epoch_mins, epoch_secs = utils.epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            cpt_epoch_no_improvement = 0
            best_valid_loss = valid_loss
            if not is_optimization:
                if is_fine_tune:
                    torch.save(model.auto_encoder.state_dict(), path_model)
                else:
                    torch.save(model.state_dict(), path_model)

        if cpt_epoch_no_improvement == early_stopping:
            if not is_optimization:
                print("Stopping earlier because no improvement")
                if is_fine_tune:
                    model.auto_encoder.load_state_dict(torch.load(path_model))
                else:
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

