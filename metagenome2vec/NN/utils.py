import os
import json
import torch
import numpy as np
from metagenome2vec.utils.string_names import *
from metagenome2vec.utils import file_manager
import ray
from ray import tune
from ax.service.ax_client import AxClient
from ray.tune.suggest.ax import AxSearch
from ray.tune.schedulers import AsyncHyperBandScheduler


def set_device(id_gpu):
    id_gpu = [int(x) for x in id_gpu.split(',')]
    if id_gpu != [-1]:
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(x) for x in id_gpu])
    return torch.device("cuda:%s" % id_gpu[0] if id_gpu != [-1] else "cpu")

def get_features(X):
    return [count_name] + [col for col in X.columns.values if str(col).isdigit()]


def ray_hyperparameter_search(path_model, path_results, train_evaluate, D_resource, num_samples=10, random_seed=42):
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
                    local_dir=os.path.join(path_results)
                    )
    file_manager.create_dir(path_model, mode="local")
    analyse.dataframe().to_csv(os.path.join(path_model, "tuning_results.csv"), index=False)
    best_parameters = analyse.get_best_config(metric=metric_to_tune, mode="max")
    with open(os.path.join(path_model, 'best_parameters'), 'w') as fp:
        json.dump(best_parameters, fp)