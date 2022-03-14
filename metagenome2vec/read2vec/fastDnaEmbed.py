import pandas as pd
import subprocess
import os
from pyspark.sql import types as T
from pyspark.sql import Window
from pyspark.sql.functions import monotonically_increasing_id, row_number, udf
import string
import random
import time
random.seed(time.time())
from metagenome2vec.read2vec.read2vec import Read2Vec
from metagenome2vec.utils import spark_manager
from metagenome2vec.utils.string_names import *


class FastDnaEmbed(Read2Vec):
    def __init__(self, path_read2genome, spark, path_tmp_folder):
        """
        :param path_read2genome: str
        :param spark: sparkContext
        :param path_tmp_folder: str
        """
        Read2Vec.__init__(self, None, None, None, None)
        self.path_fastDNA = os.getenv("FASTDNA")
        assert self.path_fastDNA is not None, "FASTDNA environment variable has to be defined"
        assert path_read2genome is not None and spark is not None and path_tmp_folder is not None, "All parameters have to be set"
        self.path_read2genome = path_read2genome
        self.spark = spark
        self.path_tmp_folder = path_tmp_folder

    def read2vec(self, X, col_name="read", drop_col_name=True):
        """
        Transform a read to a vector embedding
        Or a list of reads to a matrix embeddings
        Or a DataFrame one column "read" to a dataframe of embeddings
        :param X: Pyspark DataFrame with a read column containing a string of nucleotides
        :return: Pyspark Dataframe with embeddings rather than read
        """
        random_string = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
        while os.path.exists(os.path.join(self.path_tmp_folder, "%s_read.csv" % random_string)):
            random_string = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
        read_name = os.path.join(self.path_tmp_folder, "%s_read.csv" % random_string)
        emb_name = os.path.join(self.path_tmp_folder, "%s_emb.csv" % random_string)
        tmp_name = os.path.join(self.path_tmp_folder, "%s_tmp" % random_string)
        nun_partitions = X.rdd.getNumPartitions()
        index_name = "row_id"
        # prepare file for fastdna
        X = X.withColumn(index_name, row_number().over(Window.orderBy(monotonically_increasing_id())))
        def format_read(read, index):
            return ">" + str(index) + "\n" + str(read)

        udfFormatRead = udf(format_read, T.StringType())
        X = X.withColumn(read_formatted_name, udfFormatRead(X[col_name], X[index_name])).persist()
        X.count()

        spark_manager.write_csv_from_spark(X.select(read_formatted_name), read_name)
        # Run fastDNA
        subprocess.check_output("%s/fastdna print-word-vectors %s < %s > %s" % (self.path_fastDNA,
                                                                                self.path_read2genome,
                                                                                read_name,
                                                                                emb_name), shell=True)

        # Merge X and embedings
        sep = ' '
        X = X.drop(index_name).drop(read_formatted_name)
        spark_manager.write_csv_from_spark(X, read_name, sep=sep, mode="overwrite")
        subprocess.call("paste -d '%s' %s %s > %s" % (sep, read_name, emb_name, tmp_name), shell=True)

        n_dim = pd.read_csv(emb_name, sep=sep, nrows=1).shape[1]
        schema = T.StructType(list(X.schema) + [T.StructField(str(i), T.DoubleType(), True) for i in range(n_dim)])
        # remove last embeddings always null, repartition the dataframe and persist
        X = self.spark.read.csv(tmp_name, sep=sep, schema=schema, header=True).drop(str(n_dim-1)).repartition(nun_partitions).persist()
        X.count()

        if drop_col_name:
            X = X.drop(col_name)
        subprocess.call(["rm", "-r", read_name, emb_name, tmp_name])
        return X





