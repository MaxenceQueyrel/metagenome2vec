import torch
from torch import optim
from torch.utils.data import DataLoader

import numpy as np

SEED = 52
np.random.seed(SEED)
import os

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.svm import SVC
from sklearn.model_selection import (
    StratifiedShuffleSplit,
    RandomizedSearchCV,
    cross_validate,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from skbio.stats.composition import clr, ilr, alr, multiplicative_replacement
from sklearn.model_selection import PredefinedSplit
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm
import time
from metagenome2vec.utils import data_manager
from metagenome2vec.utils.string_names import *
from metagenome2vec.NN import vae, snn
from metagenome2vec.NN.utils import load_model, set_device
from metagenome2vec.NN.data import (
    mil_data_processing,
    get_features,
    train_test_split_mil,
)


def benchmark(
    path_save,
    X,
    y_,
    dataset_name,
    n_splits,
    test_size,
    n_iter,
    computation_type=None,
    n_cpus=1,
):
    if computation_type is not None:
        X = X / X.sum(axis=0)
        transfo = (
            clr
            if computation_type == "clr"
            else alr
            if computation_type == "alr"
            else ilr
        )
        X = transfo(multiplicative_replacement(X))

    scalar = StandardScaler()
    inner_cv = StratifiedShuffleSplit(
        n_splits=n_splits, test_size=test_size, random_state=SEED
    )
    outer_cv = StratifiedShuffleSplit(
        n_splits=n_splits, test_size=test_size, random_state=SEED
    )

    rf = Pipeline([("transformer", scalar), ("estimator", RandomForestClassifier())])
    gb = Pipeline(
        [("transformer", scalar), ("estimator", GradientBoostingClassifier())]
    )
    ab = Pipeline([("transformer", scalar), ("estimator", AdaBoostClassifier())])
    svm = Pipeline(
        [("transformer", scalar), ("estimator", SVC(max_iter=200, gamma="scale"))]
    )

    D_model = {
        rf: {
            "estimator__n_estimators": np.arange(3, 30, 1),
            "estimator__max_features": ["auto", "log2", None],
            "estimator__max_depth": [None, 3, 6, 9],
        },
        gb: {
            "estimator__learning_rate": np.random.uniform(0, 2, 30),
            "estimator__max_features": ["auto", "log2", None],
            "estimator__n_estimators": np.arange(3, 30, 1),
            "estimator__max_depth": np.arange(1, 5, 1),
        },
        ab: {
            "estimator__n_estimators": np.arange(3, 70, 1),
            "estimator__learning_rate": np.random.uniform(0, 2, 30),
        },
        svm: {
            "estimator__C": np.random.randint(1, 50, 10),
            "estimator__kernel": ["rbf", "linear", "poly"],
            "estimator__degree": np.arange(2, 30, 1),
        },
    }

    scoring = {
        "accuracy": "accuracy",
        "f1": "f1",
        "precision": "precision",
        "recall": "recall",
        "roc_auc": "roc_auc",
    }

    for model, parameters in D_model.items():
        random_search = RandomizedSearchCV(
            model,
            param_distributions=parameters,
            n_iter=n_iter,
            cv=inner_cv,
            n_jobs=n_cpus,
            scoring="accuracy",
            random_state=SEED,
        )

        # random_search.fit(X, y_)
        # clone(random_search.best_estimator_)
        scores = cross_validate(
            random_search,
            X,
            y_,
            scoring=scoring,
            cv=outer_cv,
            return_train_score=True,
            n_jobs=n_cpus,
        )
        classifier = (
            str(model[1].__class__).replace(">", "").replace("'", "").split(".")[-1]
        )
        data_manager.write_file_res_benchmarck_classif(
            path_save, dataset_name, classifier, scores
        )


class ResMetricCV:
    def __init__(self, n_splits, model_name):
        self.n_split = n_splits
        self.A_acc_train = np.zeros(n_splits)
        self.A_pre_train = np.zeros(n_splits)
        self.A_rec_train = np.zeros(n_splits)
        self.A_f1_train = np.zeros(n_splits)
        self.A_auc_train = np.zeros(n_splits)
        self.A_acc_valid = np.zeros(n_splits)
        self.A_pre_valid = np.zeros(n_splits)
        self.A_rec_valid = np.zeros(n_splits)
        self.A_f1_valid = np.zeros(n_splits)
        self.A_auc_valid = np.zeros(n_splits)
        self.A_fit_time = np.zeros(n_splits)
        self.A_score_time = np.zeros(n_splits)
        self.idx_tab = 0
        self.model_name = model_name

    def update(
        self, acc, pre, rec, f1, auc, accv, prev, recv, f1v, aucv, fit_time, score_time
    ):
        self.A_acc_train[self.idx_tab] = acc
        self.A_pre_train[self.idx_tab] = pre
        self.A_rec_train[self.idx_tab] = rec
        self.A_f1_train[self.idx_tab] = f1
        self.A_auc_train[self.idx_tab] = auc
        self.A_acc_valid[self.idx_tab] = accv
        self.A_pre_valid[self.idx_tab] = prev
        self.A_rec_valid[self.idx_tab] = recv
        self.A_f1_valid[self.idx_tab] = f1v
        self.A_auc_valid[self.idx_tab] = aucv
        self.A_fit_time[self.idx_tab] = fit_time
        self.A_score_time[self.idx_tab] = score_time
        self.idx_tab += 1

    def get_scores(self):
        scores = {
            "fit_time": self.A_fit_time,
            "score_time": self.A_score_time,
            "test_accuracy": self.A_acc_valid,
            "train_accuracy": self.A_acc_train,
            "test_f1": self.A_f1_valid,
            "train_f1": self.A_f1_train,
            "test_precision": self.A_pre_valid,
            "train_precision": self.A_pre_train,
            "test_recall": self.A_rec_valid,
            "train_recall": self.A_rec_train,
            "test_roc_auc": self.A_auc_valid,
            "train_roc_auc": self.A_auc_train,
        }
        return scores


def benchmark_with_NN(
    path_model,
    path_save,
    X,
    y_,
    model_type,
    dataset_name,
    col_features,
    n_splits,
    test_size,
    n_iter,
    n_cpus=1,
    id_gpu=-1,
    tuning=False,
    fine_tuning=False,
    add_abundance="No",
):
    nn_parameters, genomes = load_model(path_model)
    X, y_, genomes = mil_data_processing(X, y_, genomes)

    n_output = len(set(y_.tolist()))
    n_output = 1 if n_output == 2 else n_output

    if model_type == "snn":
        NN = snn.SiameseNetwork
        metagenomeDataset = snn.metagenomeDataset
        fit = snn.fit
    else:
        metagenomeDataset = vae.metagenomeDataset
        fit = vae.fit
        if model_type == "ae":
            NN = vae.AE
        elif model_type == "vae":
            NN = vae.VAE
        else:
            raise ("Not allowed computation_type: %s" % model_type)

    rf = Pipeline([("estimator", RandomForestClassifier())])
    gb = Pipeline([("estimator", GradientBoostingClassifier())])
    ab = Pipeline([("estimator", AdaBoostClassifier())])
    svm = Pipeline([("estimator", SVC(max_iter=200, gamma="scale", probability=True))])

    D_model = {
        rf: {
            "estimator__n_estimators": np.arange(3, 30, 1),
            "estimator__max_features": ["auto", "log2", None],
            "estimator__max_depth": [None, 3, 6, 9],
        },
        gb: {
            "estimator__learning_rate": np.random.uniform(0, 2, 30),
            "estimator__max_features": ["auto", "log2", None],
            "estimator__n_estimators": np.arange(3, 30, 1),
            "estimator__max_depth": np.arange(1, 5, 1),
        },
        ab: {
            "estimator__n_estimators": np.arange(3, 70, 1),
            "estimator__learning_rate": np.random.uniform(0, 2, 30),
        },
        svm: {
            "estimator__C": np.random.randint(1, 50, 10),
            "estimator__kernel": ["rbf", "linear", "poly"],
            "estimator__degree": np.arange(2, 30, 1),
        },
    }

    def score(y_, y_pred, y_prob):
        acc = accuracy_score(y_, y_pred)
        f1 = f1_score(y_, y_pred)
        pre = precision_score(y_, y_pred)
        rec = recall_score(y_, y_pred)
        auc = roc_auc_score(y_, y_prob[:, 1])
        return acc, pre, rec, f1, auc

    res_rf = ResMetricCV(n_splits, "RandomForestClassifier")
    res_svm = ResMetricCV(n_splits, "SVC")
    res_gb = ResMetricCV(n_splits, "GradientBoostingClassifier")
    res_ada = ResMetricCV(n_splits, "AdaBoostClassifier")

    col_features = get_features(X)
    device = set_device(id_gpu)

    for i, (X_train, X_valid, y_train, y_valid) in enumerate(
        tqdm(
            train_test_split_mil(
                X, y_, col_features, n_splits=n_splits, test_size=test_size
            ),
            total=n_splits,
        )
    ):
        time_transformation_with_vae = time.time()
        # Training VAE
        dataset_train = metagenomeDataset(X_train, y_train)
        dataset_valid = metagenomeDataset(X_valid, y_valid)
        params_loader = {"batch_size": nn_parameters[vae.batch_size], "shuffle": True}
        loader_train = DataLoader(dataset_train, **params_loader)
        loader_valid = DataLoader(dataset_valid, **params_loader)

        auto_encoder = NN(
            nn_parameters[vae.input_dim],
            hidden_dim=nn_parameters[vae.hidden_dim],
            n_layer_before_flatten=nn_parameters[vae.n_layer_before_flatten],
            n_layer_after_flatten=nn_parameters[vae.n_layer_after_flatten],
            device=device,
        ).to(device)

        if tuning:
            optimizer = optim.Adam(
                auto_encoder.parameters(),
                lr=nn_parameters[vae.learning_rate],
                weight_decay=nn_parameters[vae.weight_decay],
            )
            # fitting
            fit(
                auto_encoder,
                loader_train,
                loader_valid,
                optimizer,
                nn_parameters[vae.n_epoch],
                nn_parameters[vae.clip],
                path_model=path_model,
                name_model=".checkpoint.pt",
            )
            auto_encoder.eval()
        else:
            auto_encoder.load_state_dict(
                torch.load(os.path.join(path_model, vae.vae_name))
            )
            auto_encoder.eval()

        if fine_tuning:
            n_epoch = 30
            fine_tune_auto_encoder = vae.FineTuner(auto_encoder, n_output).to(device)
            optimizer = optim.Adam(
                fine_tune_auto_encoder.parameters(),
                lr=nn_parameters[vae.learning_rate],
                weight_decay=nn_parameters[vae.weight_decay],
            )
            vae.fit(
                fine_tune_auto_encoder,
                loader_train,
                loader_valid,
                optimizer,
                n_epoch,
                path_model=path_model,
            )
            auto_encoder.eval()

        # Transformed X to embeddings
        (
            X_train_transformed,
            y_train_transformed,
        ) = auto_encoder.create_matrix_embeddings(loader_train, add_abundance)
        (
            X_valid_transformed,
            y_valid_transformed,
        ) = auto_encoder.create_matrix_embeddings(loader_valid, add_abundance)
        time_transformation_with_vae = time.time() - time_transformation_with_vae

        # Training and scoring classifier
        X_transformed = np.concatenate([X_train_transformed, X_valid_transformed])
        y_transformed = np.concatenate([y_train_transformed, y_valid_transformed])

        index_cv = np.concatenate(
            [
                np.full(y_train_transformed.shape, -1),
                np.full(y_valid_transformed.shape, 0),
            ]
        )
        cv = PredefinedSplit(index_cv)

        for model, parameters in D_model.items():
            classifier = str(model[0].__class__.__name__)
            random_search = RandomizedSearchCV(
                model,
                param_distributions=parameters,
                n_iter=n_iter,
                cv=cv,
                n_jobs=n_cpus,
                scoring="accuracy",
                random_state=SEED,
                refit=False,
            )
            # Taining
            time_training_classifier = time.time()
            random_search.fit(X_transformed, y_transformed)
            model[0].set_params(
                **{
                    param.replace("estimator__", ""): val
                    for param, val in random_search.best_params_.items()
                }
            )
            model.fit(X_train_transformed, y_train_transformed)
            time_training_classifier = time.time() - time_training_classifier
            # Prediction
            time_prediction_classifier = time.time()
            y_pred_valid = model.predict(X_valid_transformed)
            time_prediction_classifier = time.time() - time_prediction_classifier
            y_prob_valid = model.predict_proba(X_valid_transformed)
            y_pred_train = model.predict(X_train_transformed)
            y_prob_train = model.predict_proba(X_train_transformed)
            # Scoring
            acc, pre, rec, f1, auc = score(
                y_train_transformed, y_pred_train, y_prob_train
            )
            accv, prev, recv, f1v, aucv = score(
                y_valid_transformed, y_pred_valid, y_prob_valid
            )
            fit_time = time_transformation_with_vae + time_training_classifier
            score_parameters = [
                acc,
                pre,
                rec,
                f1,
                auc,
                accv,
                prev,
                recv,
                f1v,
                aucv,
                fit_time,
                time_prediction_classifier,
            ]
            if classifier == "GradientBoostingClassifier":
                res_gb.update(*score_parameters)
            elif classifier == "RandomForestClassifier":
                res_rf.update(*score_parameters)
            elif classifier == "AdaBoostClassifier":
                res_ada.update(*score_parameters)
            elif classifier == "SVC":
                res_svm.update(*score_parameters)

    for res in [res_svm, res_rf, res_gb, res_ada]:
        scores = res.get_scores()
        classifier = res.model_name
        data_manager.write_file_res_benchmarck_classif(
            path_save, dataset_name, classifier, scores
        )
