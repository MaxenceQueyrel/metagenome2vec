import torch
from torch.utils.data import Dataset
from metagenome2vec.utils.string_names import *
import pandas as pd
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