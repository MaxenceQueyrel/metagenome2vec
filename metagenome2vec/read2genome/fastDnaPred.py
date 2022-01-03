from read2genome import Read2Genome
import pandas as pd
import subprocess
import os
import string
import random
import time
from pyspark.sql.functions import monotonically_increasing_id, col, row_number, udf
from pyspark.sql import types as T
from pyspark.sql import Window
from string_names import *

root_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
import hdfs_functions as hdfs

random.seed(time.time())


class FastDnaPred(Read2Genome):
    def __init__(self, path_model, path_tmp_folder):
        Read2Genome.__init__(self, "fastDNA")
        self.path_fastDNA = os.getenv("FASTDNA")
        assert self.path_fastDNA is not None, "FASTDNA environment variable has to be defined"
        assert path_model is not None and path_tmp_folder is not None, "All parameters have to be set"
        self.path_model = path_model
        self.path_tmp_folder = path_tmp_folder

    def read2genome(self, X):
        """
        transform kmers into a read embedding
        :param X: pyspark DataFrame
        :return: pandas DataFrame with prediction
        """
        col_name = "read"
        index_name = "row_id"
        random_string = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
        while os.path.exists(os.path.join(self.path_tmp_folder, "%s_read.csv" % random_string)):
            random_string = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
        read_name = os.path.join(self.path_tmp_folder, "%s_read.csv" % random_string)
        pred_name = os.path.join(self.path_tmp_folder, "%s_pred.csv" % random_string)

        # prepare file for fastdna
        X = X.withColumn(index_name, row_number().over(Window.orderBy(monotonically_increasing_id())))

        def format_read(read, index):
            return ">" + str(index) + "\n" + str(read)

        udfFormatRead = udf(format_read, T.StringType())
        X = X.withColumn(read_formatted_name, udfFormatRead(X[col_name], X[index_name])).persist()
        X.count()

        hdfs.write_csv_from_spark(X.select(read_formatted_name), read_name)

        subprocess.check_output("%s predict-prob %s %s > %s" % (os.path.join(self.path_fastDNA, "fastdna"),
                                                                self.path_model,
                                                                read_name,
                                                                pred_name), shell=True)

        preds = pd.read_csv(pred_name, header=None, sep=" ").rename(columns={0: "predict", 1: "prob"})
        X = X.drop(index_name).drop(read_formatted_name)
        X = X.select(self.getNotFeatures(X.columns)).toPandas()
        X = pd.concat([X, preds], axis=1)
        subprocess.call(["rm", "-r", read_name, pred_name])

        return X

