import time
import os
import json
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
import numpy as np
import pickle
from metagenome2vec.utils.string_names import *
from metagenome2vec.utils import file_manager
from metagenome2vec.NN.deepsets import Phi, Rho, DeepSets, score as score_deepsets, fit as fit_deepsets, table_prediction
from metagenome2vec.NN.vae import VAE, AE, fit as fit_ae, evaluate as evaluate_ae
from metagenome2vec.NN.snn import SiameseNetwork, fit as fit_snn, evaluate as evaluate_snn

from metagenome2vec.NN.data import MetagenomeNNDataset, MetagenomeSNNDataset, collate_fn, train_test_split_mil
import ray
from ray import tune
import logging
logger = logging.getLogger(tune.__name__)
logger.setLevel(level=logging.CRITICAL)
from ax.service.ax_client import AxClient
from ray.tune.suggest.ax import AxSearch
from ray.tune.schedulers import AsyncHyperBandScheduler
import logging
from tqdm import tqdm


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


def set_device(id_gpu):
    id_gpu = [int(x) for x in id_gpu.split(',')]
    if id_gpu != [-1]:
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(x) for x in id_gpu])
    return torch.device("cuda:%s" % id_gpu[0] if id_gpu != [-1] else "cpu")


def train_evaluate_factory(X, y_, model_type, input_dim, output_size, cv=10, test_size=0.2, device=torch.device("cpu")):
    def train_evaluate_decorator(function):
        def wrapper(parameterization):
            mean_accuracy, std_accuracy = function(X, y_, model_type, parameterization, input_dim=input_dim, output_size=output_size,
                                                    cv=cv, test_size=test_size, device=device, is_optimization=True)
            tune.report(mean_accuracy=mean_accuracy, std_accuracy=std_accuracy)
        return wrapper
    return train_evaluate_decorator


def ray_hyperparameter_search(X, y_, model_type, optimization_function, path_model, input_dim, D_resource, output_size=None, num_samples=10, cv=10, test_size=0.2,
                            device=torch.device("cpu"), random_seed=42):
    assert model_type in ["deepsets", "ae", "vae", "snn"]
    os.environ["RAY_PICKLE_VERBOSE_DEBUG"] = "1"
    ray.init(num_cpus=np.int(np.ceil(D_resource["cpu"] * D_resource["worker"])),
        num_gpus=np.int(np.ceil(D_resource["gpu"] * D_resource["worker"])))
    parameters = [{"name": learning_rate, "type": "range", "bounds": [1e-5, 1e-2], "log_scale": True},
                  {"name": weight_decay, "type": "fixed", "value": 0.0002},
                  {"name": dropout, "type": "choice", "values": [0.0, 0.3]},
                  {"name": batch_size, "type": "fixed", "value": 6},
                  {"name": n_epoch, "type": "fixed", "value": 100},
                  {"name": clip, "type": "fixed", "value": -1.}]
    if model_type == "deepsets":
        metric_to_tune = "mean_accuracy"
        minimize = False
        parameters.append({"name": n_layer_phi, "type": "choice", "values": [1, 3]})
        parameters.append({"name": n_layer_rho, "type": "choice", "values": [1, 3]})
        parameters.append({"name": hidden_init_phi, "type": "choice", "values": [50, 100]})
        parameters.append({"name": hidden_init_rho, "type": "choice", "values": [50, 100]})
        parameters.append({"name": mil_layer, "type": "choice", "values": ["sum", "attention"]})
    else:  # vae, ae or snn
        metric_to_tune = "mean_score"
        minimize = True
        parameters.append({"name": activation_function, "type": "choice", "values": ["nn.ReLU", "nn.LeakyReLU"], "value_type": "str"})
        parameters.append({"name": n_layer_before_flatten, "type": "choice", "values": [2, 3, 4], "value_type": "int"})
        parameters.append({"name": n_layer_after_flatten, "type": "choice", "values": [2, 3, 4], "value_type": "int"})
        parameters.append({"name": hidden_dim, "type": "choice", "values": [30, 50], "value_type": "int"})

    resources_per_trial = {"cpu": D_resource["cpu"], "gpu": D_resource["gpu"]}
    ax = AxClient(enforce_sequential_optimization=False, random_seed=random_seed)
    
    ax.create_experiment(
        name="deepsets_experiment",
        parameters=parameters,
        objective_name=metric_to_tune,
        minimize=minimize
    )

    algo = AxSearch(ax_client=ax, max_concurrent=D_resource["worker"])
    scheduler = AsyncHyperBandScheduler()

    train_evaluate = train_evaluate_factory(X=X, y_=y_, model_type=model_type, input_dim=input_dim, output_size=output_size,
                                            cv=cv, test_size=test_size, device=device)(optimization_function)

    file_manager.create_dir(path_model, mode="local")
    analyse = tune.run(train_evaluate,
                    num_samples=num_samples,
                    search_alg=algo,
                    scheduler=scheduler,
                    mode="max",
                    metric=metric_to_tune,
                    verbose=1,
                    resources_per_trial=resources_per_trial,
                    local_dir=os.path.join(path_model, 'ray_tune_results')
                    )
    
    analyse.dataframe().to_csv(os.path.join(path_model, "tuning_results.csv"), index=False)
    best_parameters = analyse.get_best_config(metric=metric_to_tune, mode="max")
    with open(os.path.join(path_model, 'best_parameters'), 'w') as fp:
        json.dump(best_parameters, fp)


def load_and_update_parameters(path_parameters, default_parameters=None):
    if os.path.exists(path_parameters):
        logging.info("Best parameters used")
        with open(path_parameters, 'r') as fp:
            parameters = json.load(fp)
        if default_parameters:
            # Change the parameters with the best ones
            for k, v in parameters.items():
                default_parameters[k] = v
            return default_parameters
        return parameters
    else:
        logging.info("Default parameters used")
        if default_parameters:
            return default_parameters
        return {}


def cross_val_score(X, y_, model_type, params, input_dim, output_size=None, cv=10, 
                    test_size=0.2, device=torch.device("cpu"), path_model="./model.pt", is_optimization=False):
    best_acc = best_score = np.inf
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
    A_score_train = np.zeros(cv)
    A_score_valid = np.zeros(cv)
    for i, (X_train, X_valid, y_train, y_valid) in enumerate(tqdm(train_test_split_mil(X, y_, n_splits=cv, test_size=test_size), total=cv)):
        params_loader = {'batch_size': params[batch_size],
                         'shuffle': True}
        # Create the model
        if model_type == "deepsets":
            assert output_size is not None, "output_size must be define for deepsets model"
            phi = Phi(input_dim, params[hidden_init_phi], params[n_layer_phi], params[dropout])
            rho = Rho(phi.last_hidden_size, params[hidden_init_rho], params[n_layer_rho], params[dropout], output_size)
            model = DeepSets(phi, rho, params[mil_layer], device).to(device)
            # Create the datasets
            dataset_train = MetagenomeNNDataset(X_train, y_train)
            dataset_valid = MetagenomeNNDataset(X_valid, y_valid)
            params_loader["collate_fn"] = lambda x: collate_fn(x, device, batch_format=False)
        elif model_type == "vae" or model_type == "ae":
            NN = VAE if model_type == "vae" else AE
            model = NN(input_dim=input_dim, hidden_dim=params[hidden_dim],
                   n_layer_before_flatten=params[n_layer_before_flatten],
                   n_layer_after_flatten=params[n_layer_after_flatten],
                   device=device, activation_function=params[activation_function]).to(device)
            # Create the datasets
            dataset_train = MetagenomeNNDataset(X_train, y_train)
            dataset_valid = MetagenomeNNDataset(X_valid, y_valid)
            params_loader["collate_fn"] = lambda x: collate_fn(x, device, batch_format=True)
        elif model_type == "snn":
            model = SiameseNetwork(input_dim=input_dim, hidden_dim=params[hidden_dim],
                   n_layer_before_flatten=params[n_layer_before_flatten],
                   n_layer_after_flatten=params[n_layer_after_flatten],
                   device=device, activation_function=params[activation_function]).to(device)
            # Create the datasets
            dataset_train = MetagenomeSNNDataset(X_train, y_train)
            dataset_valid = MetagenomeSNNDataset(X_valid, y_valid)
     
        loader_train = DataLoader(dataset_train, **params_loader)
        loader_valid = DataLoader(dataset_valid, **params_loader)
        
        optimizer = optim.Adam(model.parameters(), lr=params[learning_rate], weight_decay=params[weight_decay])
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.3)

        d = time.time()
        # DeepSets
        if model_type == "deepsets":
            fit_deepsets(model, loader_train, loader_valid, optimizer, params[n_epoch],
                params[clip], scheduler, path_model=path_model, is_optimization=is_optimization)
            A_fit_time[i] = time.time() - d
            d = time.time()
            acc, pre, rec, f1, auc = score_deepsets(model, loader_train)
            A_acc_train[i], A_pre_train[i], A_rec_train[i], A_f1_train[i], A_auc_train[i]  = acc, pre, rec, f1, auc
            # scoring on valid
            acc, pre, rec, f1, auc = score_deepsets(model, loader_valid)
            A_acc_valid[i], A_pre_valid[i], A_rec_valid[i], A_f1_valid[i], A_auc_valid[i]  = acc, pre, rec, f1, auc
            A_score_time[i] = time.time() - d
            if best_acc < acc:
                torch.save(model.state_dict(), path_model)
                if params[mil_layer] == "attention":
                    path_model_parent = os.path.dirname(path_model)
                    table_prediction(model, dataset_train).to_csv(os.path.join(path_model_parent, "train_best_model.csv"), index=False)
                    table_prediction(model, dataset_valid).to_csv(os.path.join(path_model_parent, "valid_best_model.csv"), index=False)
            if is_optimization:
                return np.mean(A_acc_valid), np.std(A_acc_valid)
            return {"test_accuracy": A_acc_valid,
                "fit_time": A_fit_time, "score_time": A_score_time,
                "test_accuracy": A_acc_valid, "train_accuracy": A_acc_train,
                "test_f1": A_f1_valid, "train_f1": A_f1_train,
                "test_precision": A_pre_valid, "train_precision": A_pre_train,
                "test_recall": A_rec_valid, "train_recall": A_rec_train,
                "test_roc_auc": A_auc_valid, "train_roc_auc": A_auc_train}
        # Auto Encoder
        elif model_type == "vae" or model_type == "ae":
            fit_ae(model, loader_train, loader_valid, optimizer, params[n_epoch], params[clip],
                scheduler, path_model=path_model, is_optimization=is_optimization)
            A_fit_time[i] = time.time() - d
            d = time.time()
            score_train = evaluate_ae(model, loader_train)
            score_valid = evaluate_ae(model, loader_train)
            A_score_train[i], A_score_valid[i] = score_train, score_valid
            A_score_time[i] = time.time() - d
            if score_valid < best_score:
                best_score = score_valid
                torch.save(model.state_dict(), path_model)
            if is_optimization:
                return np.mean(A_score_valid), np.std(A_score_valid)
            return {"score_train": A_score_train,
                "score_valid": A_score_valid,
                "fit_time": A_fit_time, "score_time": A_score_time}
        elif model_type == "snn":
            fit_snn(model, loader_train, loader_valid, optimizer, params[n_epoch], params[clip], 
                    scheduler, path_model=path_model)
            A_fit_time[i] = time.time() - d
            d = time.time()
            score_train = evaluate_snn(model, loader_train)
            score_valid = evaluate_snn(model, loader_train)
            A_score_train[i], A_score_valid[i] = score_train, score_valid
            A_score_time[i] = time.time() - d
            if score_valid < best_score:
                best_score = score_valid
                torch.save(model.state_dict(), path_model)
            if is_optimization:
                return np.mean(A_score_valid), np.std(A_score_valid)
            return {"score_train": score_train,
                "score_valid": score_valid,
                "fit_time": A_fit_time, "score_time": A_score_time}
        else:
            raise "%s model type does not exist" % model_type

# TODO use this function
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
