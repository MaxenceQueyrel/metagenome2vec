import time
import os
import json
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
import numpy as np
from metagenome2vec.utils.string_names import *
from metagenome2vec.utils import file_manager
from metagenome2vec.NN.deepsets import Phi, Rho, DeepSets, score, fit, table_prediction
from metagenome2vec.data_processing.metagenomeNNDataset import MetagenomeNNDataset, collate_fn, train_test_split_mil
import ray
from ray import tune
from ax.service.ax_client import AxClient
from ray.tune.suggest.ax import AxSearch
from ray.tune.schedulers import AsyncHyperBandScheduler
import logging
from tqdm import tqdm

def set_device(id_gpu):
    id_gpu = [int(x) for x in id_gpu.split(',')]
    if id_gpu != [-1]:
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(x) for x in id_gpu])
    return torch.device("cuda:%s" % id_gpu[0] if id_gpu != [-1] else "cpu")


def train_evaluate_factory(X, y_, model_type, embed_size, output_size, cv=10, test_size=0.2, device=torch.device("cpu")):
    def train_evaluate_decorator(function):
        def wrapper(parameterization):
            mean_accuracy, std_accuracy = function(X, y_, model_type, parameterization, embed_size=embed_size, output_size=output_size,
                                                    cv=cv, test_size=test_size, device=device, is_optimization=True)
            tune.report(mean_accuracy=mean_accuracy, std_accuracy=std_accuracy)
        return wrapper
    return train_evaluate_decorator


def ray_hyperparameter_search(X, y_, model_type, optimization_function, path_model, embed_size, output_size, D_resource, num_samples=10, cv=10, test_size=0.2,
                             minimize=False, device=torch.device("cpu"), random_seed=42):
    ray.init(num_cpus=np.int(np.ceil(D_resource["cpu"] * D_resource["worker"])),
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
    
    resources_per_trial = {"cpu": D_resource["cpu"], "gpu": D_resource["gpu"]}
    ax = AxClient(enforce_sequential_optimization=False, random_seed=random_seed)
    metric_to_tune = "mean_accuracy"
    ax.create_experiment(
        name="deepsets_experiment",
        parameters=parameters,
        objective_name=metric_to_tune,
        minimize=minimize
    )

    algo = AxSearch(ax_client=ax, max_concurrent=D_resource["worker"])
    scheduler = AsyncHyperBandScheduler()

    train_evaluate = train_evaluate_factory(X=X, y_=y_, model_type=model_type, embed_size=embed_size, output_size=output_size, cv=cv, test_size=test_size, device=device)(optimization_function)

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


def load_best_parameters(path_best_parameters):
    if os.path.exists(path_best_parameters):
        logging.info("Best parameters used")
        with open(path_best_parameters, 'r') as fp:
            return json.load(fp)
    else:
        logging.info("Default parameters used")
        return {}


def cross_val_score(X, y_, model_type, params, embed_size, output_size, cv=10, 
                    test_size=0.2, device=torch.device("cpu"), path_model="./deepsets.pt", is_optimization=False):
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
        if model_type == "deepsets":
            phi = Phi(embed_size, params[hidden_init_phi], params[n_layer_phi], params[dropout])
            rho = Rho(phi.last_hidden_size, params[hidden_init_rho], params[n_layer_rho], params[dropout], output_size)
            model = DeepSets(phi, rho, params[mil_layer], device).to(device)
            if output_size <= 2:
                criterion = nn.BCEWithLogitsLoss()
            else:
                criterion = nn.CrossEntropyLoss()
        
        optimizer = optim.Adam(model.parameters(), lr=params[learning_rate], weight_decay=params[weight_decay])
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.3)
        
        # fitting
        d = time.time()
        model = fit(model, loader_train, loader_valid, optimizer, criterion, params[n_epoch],
                    params[clip], scheduler, path_model=path_model, is_optimization=is_optimization)
        A_fit_time[i] = time.time() - d
        # scoring on train
        d = time.time()
        acc, pre, rec, f1, auc = score(model, loader_train)
        A_acc_train[i] = acc
        A_pre_train[i] = pre
        A_rec_train[i] = rec
        A_f1_train[i] = f1
        A_auc_train[i] = auc
        # scoring on valid
        acc, pre, rec, f1, auc = score(model, loader_valid)
        A_score_time[i] = time.time() - d
        A_acc_valid[i] = acc
        A_pre_valid[i] = pre
        A_rec_valid[i] = rec
        A_f1_valid[i] = f1
        A_auc_valid[i] = auc
        if model_type == "deepsets" and best_acc < acc:
            best_acc = acc
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
