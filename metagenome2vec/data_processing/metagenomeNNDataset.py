import torch
from torch.utils.data import Dataset
from metagenome2vec.utils.string_names import *
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler


class MetagenomeNNDataset(Dataset):
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
        idx2 = idx
        while idx2 == idx:
            idx2 = np.random.choice(len(self), 1)
        data1, label1 = self.data[idx], self.labels[idx]
        data2, label2 = self.data[idx2], self.labels[idx2]
        return data1, data2, int(label1 == label2), label1


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


def get_features(X):
    return [count_name] + [col for col in X.columns.values if str(col).isdigit()]


# TODO check  both function train_test_split
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


def mil_data_processing(X, y_, genomes=None):
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
    # Reform the original dataset and order by id and genome
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