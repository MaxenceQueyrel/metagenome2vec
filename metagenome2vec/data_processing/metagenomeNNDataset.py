from torch.utils.data import Dataset
from metagenome2vec.utils.string_names import *
import pandas as pd


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