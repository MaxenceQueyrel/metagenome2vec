import numpy as np
from metagenome2vec.read2vec.read2vec import Read2Vec
import fasttext


class FastTextReadEmbeddings(Read2Vec):
    def __init__(self, path_model, dico_index, reverse_index, k_size):
        """
        :param self: Read2Vec object
        :param path_model: str, path to fasttext model
        :param dico_index: dictionary with the kmer and its index in the embeddings matrix
        :param reverse_index: dictionary with index as key and kmer as value
        :param k_size: int, kmer size
        """
        Read2Vec.__init__(self, dico_index, reverse_index, k_size, to_int=False)
        self.path_model = path_model

    def _transform(self, x):
        """
        transform kmers into a read embedding
        :param x: str, tring of k-mers
        :return: 1-D Numpy array
        """
        return fasttext.load_model(self.path_model).get_sentence_vector(x)

    def transform_all(self, X):
        """
        Transform a matrix of kmer idx (one line a read, one column a kmer) into a matrix of reads embeddings
        => one line a read one column a dimension of an embedding
        :param X: Numpy 2-D array
        :return: Numpy 2-D array
        """
        model = fasttext.load_model(self.path_model)
        vfunc = np.vectorize(lambda x: model.get_sentence_vector(x), signature='()->(n)')
        return vfunc(X)

