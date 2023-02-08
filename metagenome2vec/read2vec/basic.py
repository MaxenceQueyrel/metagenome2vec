import numpy as np
from metagenome2vec.read2vec.read2vec import Read2Vec


class BasicReadEmbeddings(Read2Vec):
    def __init__(
        self,
        embeddings,
        dico_index,
        reverse_index,
        k_size,
        word_count=None,
        agg_func=np.mean,
    ):
        """
        :param self: Read2Vec object
        :param embeddings: numpy 2D array, the kmers embeddings
        :param dico_index: dictionary with the kmer and its index in the embeddings matrix
        :param reverse_index: dictionary with index as key and kmer as value
        :param k_size: int, kmer size
        :param word_count: dictionary with the kmer and its count in the corpus
        :param agg_func: String, np.mean or np.sum
        """
        Read2Vec.__init__(self, dico_index, reverse_index, k_size)
        self.embeddings = embeddings
        self.agg_func = agg_func
        self.tfidf = None
        if word_count is not None:
            tot = np.sum(list(word_count.values()))
            self.tfidf = np.ones(len(self.dico_index))
            for k, v in word_count.items():
                try:
                    self.tfidf[self.dico_index[k]] = np.log(1.0 / (v * 1.0 / tot))
                except:
                    continue
            # self.tfidf = {k: np.log(1./(v*1./tot)) for k, v in word_count.iteritems()}

    @staticmethod
    def l2_norm(x):
        return np.sqrt(np.sum(x**2, axis=1))

    @staticmethod
    def div_norm(x):
        norm_value = BasicReadEmbeddings.l2_norm(x)
        return np.array(
            [xx * (1.0 / nv) if nv > 0 else xx for xx, nv in zip(x, norm_value)]
        )

    def _transform(self, x):
        """
        transform kmers into a read embedding
        :param x: 1-D numpy array, index of the kmer in reverse_index (int to kmer)
        :return: 1-D Numpy array
        """
        # remove pad and unk index
        x = np.delete(
            x, np.where((x == self.index_pad) | (x < 0))[0]
        )  # Case unk kmer and "<unk>" is not defined in the vocabulary because -1 is default
        if x.shape[0] == 0:
            return np.zeros(self.embeddings.shape[1])
        if self.tfidf is not None:
            res = self.div_norm(self.embeddings[x]) * self.tfidf[x][:, np.newaxis]
        else:
            res = self.div_norm(self.embeddings[x])
        return self.agg_func(res, axis=0)

    def transform_all(self, X):
        """
        Transform a matrix of kmer idx (one line a read, one column a kmer) into a matrix of reads embeddings
        => one line a read one column a dimension of an embedding
        :param X: Numpy 2-D array
        :return: Numpy 2-D array
        """
        return np.apply_along_axis(lambda x: self._transform(x), axis=1, arr=X)
