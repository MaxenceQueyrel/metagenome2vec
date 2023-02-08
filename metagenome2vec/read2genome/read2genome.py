from abc import abstractmethod


class Read2Genome(object):
    def __init__(self, name):
        self.name = name

    def read2genome(self, df):
        """
        transform kmers into a read embedding
        :param df: pyspark DataFrame
        :return: pyspark Dataframe with prediction
        """
        return NotImplemented

    @staticmethod
    def getFeatures(columns):
        """
        :param columns: List of str
        :return:
        """
        features = []
        for col in columns:
            try:
                a = int(col)
                features.append(str(a))
            except:
                continue
        return features

    @staticmethod
    def getNotFeatures(columns):
        """
        :param columns: List of str
        :return:
        """
        notFeatures = []
        for col in columns:
            try:
                int(col)
            except:
                notFeatures.append(col)
        return notFeatures
