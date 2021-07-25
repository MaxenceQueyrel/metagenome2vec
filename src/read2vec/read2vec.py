import sys
import os
from collections import Counter, defaultdict
import pyspark.sql.types as T
import pyspark.sql.functions as F
import numpy as np
import pandas as pd
from tqdm import tqdm
import abc

root_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, os.path.join(root_folder, "utils"))

if sys.version_info[0] == 3 and sys.version_info[1] == 7:
    import transformation_ADN
else:
    import transformation_ADN2 as transformation_ADN

#os.environ['ARROW_PRE_0_15_IPC_FORMAT'] = '1'


class Read2Vec(object):
    def __init__(self, dico_index, reverse_index, k_size, max_length=-1, to_int=True):
        """
        :param dico_index: dictionary with the kmer and its index in the embeddings matrix
        :param reverse_index: dictionary with the index as key and the kmer as value
        :param k_size: int, kmer size
        :param max_length: int, default -1, length of the sequence, if -1 no limit
        """
        self.dico_index = dico_index
        self.reverse_index = reverse_index
        self.k_size = k_size
        # index_unk: The index for kmers unknown
        # index_pad: The index for padding
        self.index_unk = -1
        # If <unk> kmer is defined then change the real index else all unkowns words will be exculde
        if self.dico_index is not None and "<unk>" in self.dico_index:
            self.index_unk = self.dico_index["<unk>"]
        self.index_pad = -2
        self.max_length = max_length
        self.to_int = to_int

    @abc.abstractmethod
    def _transform(self, x):
        """
        transform kmers into a read embedding
        :param x: 1-D numpy array, index of the kmer in reverse_index (int to kmer)
        :return: 1-D Numpy array
        """
        raise NotImplementedError("Not implemented")

    def transform(self, x, tolist=True):
        """
        transform kmers into a read embedding
        :param x: 1-D numpy array, index of the kmer in reverse_index (int to kmer)
        :param tolist: Boolean, if true return a list, if false return a numpy array
        :return: 1-D Numpy array or list
        """
        res = self._transform(x)
        if tolist:
            return res.tolist()
        return res

    @abc.abstractmethod
    def transform_all(self, X):
        """
        Transform a matrix of kmer (one line a read, one column a kmer) into a matrix of reads embeddings
        => one line a read one column a dimension of an embedding
        :param X: Numpy 2-D array, index of the kmer in reverse_index (int to kmer)
        :return: Numpy 2-D array
        """
        raise NotImplementedError("Not implemented")

    def readProjectionWrapper(self):
        # case batch pandas udf
        if not getattr(self.transform_all, "__isabstractmethod__", False):
            def readProjection(L_read):
                return pd.Series([elem for elem in self.transform_all(self.preprocess_several_reads(list(L_read)))])
            return F.pandas_udf(readProjection, T.ArrayType(T.DoubleType()))
        # case normal udf
        else:
            def readProjection(read):
                return self.transform(self.preprocess_read(read))
            return F.udf(readProjection, T.ArrayType(T.DoubleType()))

    def preprocess_read(self, read):
        """
        Create all 'sentences' of feasible kmers in read
        :param read: String, DNA sequence
        :return: numpy 1D array int, idx of kmer
        """
        if self.to_int:
            return transformation_ADN.preprocess_read(read, self.k_size, self.dico_index,
                                                      self.index_unk, self.index_pad,
                                                      self.max_length)
        else:
            return transformation_ADN.preprocess_read_str(read, self.k_size, self.max_length)

    def preprocess_several_reads(self, L_read):
        """
        Create all 'sentences' of feasible kmers with the list of reads L_read
        :param L_read: list, list of DNA sequence (string)
        :return: numpy 2D array int, idx of kmer
        The first column is an index to retrieve the rows with the reads
        """
        if self.to_int:
            return transformation_ADN.preprocess_several_reads(L_read, self.k_size, self.dico_index,
                                                               self.index_unk, self.index_pad,
                                                               self.max_length)
        else:
            return transformation_ADN.preprocess_several_reads_str(L_read, self.k_size, self.max_length)

    def read2vec(self, X, col_name="read", drop_col_name=True):
        """
        Transform reads to embeddings
        :param X: Pyspark DataFrame with a column 'col' containing a string of nucleotides
        :param col_name: String, the name of the column to transform in embeddings
        :param drop_col_name: Bool, True if we want to remove col_name
        :param to_int: bool, True if transform read into list of index else transform read in string of kmers
        :return: Pyspark Dataframe with embeddings, one column by dimension
        """
        emb_col_name = "embeddings"
        X = X.withColumn(emb_col_name, self.readProjectionWrapper()(X[col_name]))
        if drop_col_name:
            X = X.drop(col_name)
        X = X.persist()
        X.count()

        emb_dim = len(X.select(emb_col_name).take(1)[0][0])
        X = X.select(*[col for col in X.columns if col != emb_col_name] + \
                      [X.embeddings[int("%s" % i)].alias("%s" % i) for i in range(0, emb_dim)]).persist()
        X.count()
        return X

    @staticmethod
    def init_stoi_itos_cutoffs(L_path_data, min_freq=5, nb_cutoffs=None):
        """
        Create stoi itos and cutoffs variables for the adaptative softmax
        :param L_path_data: List, list of several path, example one for training the other for validation
        :param: min_freq: int, Number of minimum occurrence of one kmer
        :param nb_cutoffs: Int, number of cut off for the adaptative clustering
        :return:
        """
        counter = Counter()
        for path_data in L_path_data:
            with open(path_data, 'rt') as f:
                for line in tqdm(f):
                    counter.update(line.strip().split(' '))

        stoi = defaultdict()
        for i, value in enumerate(counter.most_common()):
            stoi[value[0]] = i

        def _default_unk_index():
            return 0

        stoi.default_factory = _default_unk_index
        L_special_token = ["<unk>", "<pad>", "<sos>", "<eos>", "<msk>"]
        stoi.update((x, y + len(L_special_token)) for x, y in stoi.items())
        for i, token in enumerate(L_special_token):
            stoi[token] = i

        # Removing gab between values
        prev_v = 0
        for k in sorted(stoi, key=stoi.get):
            v = stoi[k]
            if v - prev_v > 1:
                v = prev_v + 1
                stoi.update({k: v})
            prev_v = v

        if nb_cutoffs is not None and nb_cutoffs > 0:
            # Determine cutoffs
            cs = np.cumsum(np.array(np.array(counter.most_common())[:, 1], dtype=int))
            cs = cs * 1. / cs[-1]
            pct_cut = 1. / nb_cutoffs
            cutoffs = []
            i = 1
            for nb_elem_in_cutoff, pct in enumerate(cs):
                if pct > i * pct_cut:
                    cutoffs.append(nb_elem_in_cutoff)
                    i += 1
                    if i == nb_cutoffs:
                        break
        else:
            cutoffs = None
        # cutoffs = [500, 1000, 10000, 20000]
        return stoi, [x[0] for x in sorted(list(stoi.items()), key=lambda x: x[1])], cutoffs


