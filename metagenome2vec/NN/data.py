import torch
from torch.utils.data import Dataset
from metagenome2vec.utils.string_names import *
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler


class MetagenomeNNDataset(Dataset):
    def __init__(self, X, y_):
        if isinstance(y_, pd.Series):
            y_ = y_.values 
        self.data = X.copy()
        self.data[group_name] = y_.copy()
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


class MetagenomeSNNDataset(MetagenomeNNDataset):
    def __init__(self, X, y_):
        super().__init__(X, y_)

    def __getitem__(self, idx):
        idx2 = idx
        while idx2 == idx:
            idx2 = np.random.choice(len(self), 1)[0]
        data1, label1 = self.D_data[self.IDs[idx]], self.labels[idx]
        data2, label2 = self.D_data[self.IDs[idx]], self.labels[idx]
        return data1, data2, int(label1 == label2), label1


def item_batch_to_tensor(item, device):
    d = torch.tensor(item[0], dtype=torch.float, device=device)
    t = item[1]
    g = torch.tensor(item[2], dtype=torch.int, device=device)
    return d, t, g


def collate_fn(batch, device, batch_format=False):
    data, target, genome = [], [], []
    for elem in batch:
        d, t, g = item_batch_to_tensor(elem, device)
        data.append(d)
        target.append(t)
        genome.append(g)
    if batch_format:
        data = torch.concat([x.unsqueeze(0) for x in data])
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
    X = X.copy()
    X[group_name] = y_.copy()
    col_features = get_features(X)
    X_uniq = X.drop_duplicates(subset=[id_subject_name, group_name])[[id_subject_name, group_name]]
    sss = StratifiedShuffleSplit(n_splits, test_size=test_size)
    for id_train, id_valid in sss.split(X_uniq[id_subject_name], X_uniq[group_name]):
        id_train = X[id_subject_name].isin(X_uniq[id_subject_name].iloc[id_train])
        id_valid = X[id_subject_name].isin(X_uniq[id_subject_name].iloc[id_valid])
        X_train, X_valid, y_train, y_valid = X[id_train].reset_index(drop=True).drop(group_name, axis=1), \
                                            X[id_valid].reset_index(drop=True).drop(group_name, axis=1), \
                                            y_[id_train], y_[id_valid]
        scalar = StandardScaler()
        X_train[col_features] = scalar.fit_transform(X_train[col_features])
        X_valid[col_features] = scalar.transform(X_valid[col_features])
        yield X_train, X_valid, y_train, y_valid


def load_matrix_for_learning(path_matrix, path_metadata, disease, is_bok=False, nan_to_control=True, model_type=None):
    """
    Load a matrix to feed into machine learning algorithm
    :param path_matrix: String, Complete path to the matrix formed with almost 2 columns:
        - id.fasta, String, the id of the metagenome file
        - group, String, the class value => empty for unknown (rows are removed), Control (switch to 0) and a disease name (switch to 1)
    :param: path_metadata: Path to the info about metagenome
    :param disease: String, The disease class in metadata file
    :param: is_bok: Boolean, True if it is a bok matrix, then a group by sum is applied instead of group by mean
    :param: nan_to_control: Boolean, True to transform nan into control case else delete these elements
    :return: X, Pandas 2D DataFrame and y_, numpy 1D array the vs
    """
    X = pd.read_csv(path_matrix, sep=",")
    if pair_name in X.columns.values:
        del X[pair_name]
    metadata = pd.read_csv(path_metadata, delimiter=",")[[id_fasta_name, group_name, id_subject_name]].astype(str)
    if group_name in X.columns.values:
        del metadata[group_name]
    X = X.merge(metadata, on=id_fasta_name)
    if nan_to_control:
        X.loc[X[group_name].isna(), group_name] = control_category
    X = X[~X.isnull().any(axis=1)]
    if count_name in X.columns.values:  # Case MIL
        #count = X.groupby(by=[id_subject_name, group_name, genome_name], as_index=False).agg({count_name: "sum"})[[count_name]]
        #del X[count_name]
        #X = X.groupby(by=[id_subject_name, group_name, genome_name], as_index=False).mean()
        #X = pd.concat([count, X], axis=1)
        X = X.groupby(by=[id_subject_name, group_name, genome_name], as_index=False).sum()
    else:  # case SIL
        if is_bok:
            X = X.groupby(by=[id_subject_name, group_name], as_index=False).sum()
        else:
            X = X.groupby(by=[id_subject_name, group_name], as_index=False).mean()
    y_ = np.where(np.array(X[group_name].astype(str)) == disease, 1, 0)
    del X[group_name]
    if model_type is None or model_type == "deepsets":
        return X, y_
    return mil_data_processing(X, y_)


def load_several_matrix_for_learning(path_data, path_metadata, disease, model_type=None):
    cpt_class = 2
    X = y_ = None
    for path_d, path_m, d in zip(path_data.split(","), path_metadata.split(","), disease.split(",")):
        X_tmp, y_tmp = load_matrix_for_learning(path_d, path_m, d, model_type=model_type)
        if X is None:
            X, y_ = X_tmp, y_tmp
        else:
            X = pd.concat([X, X_tmp])
            y_ = np.concatenate([y_, np.where(y_tmp == 1, cpt_class, y_tmp)])
            cpt_class += 1
    if model_type in ["ae", "vae", "snn"]:
        return mil_data_processing(X, y_)
    return X, y_


def mil_data_processing(X, y_, genomes=None):
    X[group_name] = y_
    X[genome_name] = X[genome_name].astype(int)
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
    X[genome_name] = X[genome_name].astype(int)
    return X, y_