import pandas as pd
import numpy as np
import subprocess
import os
import string
import random
import time
from pyspark.sql.functions import monotonically_increasing_id, row_number, udf
from pyspark.sql import types as T
from pyspark.sql import Window
from metagenome2vec.utils.string_names import *

from metagenome2vec.utils import spark_manager, file_manager, data_manager
from metagenome2vec.read2genome.read2genome import Read2Genome

random.seed(time.time())


class FastDnaPred(Read2Genome):
    def __init__(self, path_model=None, path_tmp_folder=None):
        Read2Genome.__init__(self, "fastDNA")
        self.path_fastDNA = os.getenv("FASTDNA")
        assert (
            self.path_fastDNA is not None
        ), "FASTDNA environment variable has to be defined"
        self.path_model = path_model
        self.path_tmp_folder = path_tmp_folder

    def read2genome(self, X):
        """
        transform kmers into a read embedding
        :param X: pyspark DataFrame
        :return: pandas DataFrame with prediction
        """
        assert (
            self.path_model is not None and self.path_tmp_folder is not None
        ), "path_model and path_tmp_folder variables have to be set."
        col_name = "read"
        index_name = "row_id"
        random_string = "".join(
            random.choice(string.ascii_uppercase + string.digits) for _ in range(10)
        )
        while os.path.exists(
            os.path.join(self.path_tmp_folder, "%s_read.csv" % random_string)
        ):
            random_string = "".join(
                random.choice(string.ascii_uppercase + string.digits) for _ in range(10)
            )
        read_name = os.path.join(self.path_tmp_folder, "%s_read.csv" % random_string)
        pred_name = os.path.join(self.path_tmp_folder, "%s_pred.csv" % random_string)

        # prepare file for fastdna
        X = X.withColumn(
            index_name, row_number().over(Window.orderBy(monotonically_increasing_id()))
        )

        def format_read(read, index):
            return ">" + str(index) + "\n" + str(read)

        udfFormatRead = udf(format_read, T.StringType())
        X = X.withColumn(
            read_formatted_name, udfFormatRead(X[col_name], X[index_name])
        ).persist()
        X.count()

        spark_manager.write_csv_from_spark(X.select(read_formatted_name), read_name)

        subprocess.check_output(
            "%s predict-prob %s %s > %s"
            % (
                os.path.join(self.path_fastDNA, "fastdna"),
                self.path_model,
                read_name,
                pred_name,
            ),
            shell=True,
        )

        preds = pd.read_csv(pred_name, header=None, sep=" ").rename(
            columns={0: "predict", 1: "prob"}
        )
        X = X.drop(index_name).drop(read_formatted_name)
        X = X.select(self.getNotFeatures(X.columns)).toPandas()
        X = pd.concat([X, preds], axis=1)
        subprocess.call(["rm", "-r", read_name, pred_name])

        return X

    def train(
        self,
        path_data,
        k_mer_size,
        embedding_size,
        n_steps,
        learning_rate,
        tax_taken,
        path_kmer2vec,
        path_read2genome,
        path_tmp_folder,
        n_cpus,
        noise,
        max_length,
    ):
        data, labels = path_data.split(",")
        tax_taken = (
            None
            if (tax_taken is None or tax_taken == "None")
            else [str(x) for x in tax_taken.split(".")]
        )

        f_name = (
            "fastdna_k_{}_embed_size_{}_n_step_{}_lr_{}_noise_{}_max_length_{}".format(
                k_mer_size, embedding_size, n_steps, learning_rate, noise, max_length
            )
        )
        if tax_taken is not None:
            f_name += "_n_tax_%s" % len(tax_taken)
        path_embedding = os.path.join(path_kmer2vec, f_name)
        path_model = os.path.join(path_read2genome, f_name)

        path_fastDNA = os.getenv("FASTDNA")
        assert (
            path_fastDNA is not None
        ), "FASTDNA environment variable has to be defined"

        ###################################
        # --------- Create folders -------#
        ###################################

        # Create the folder where the model will be saved
        file_manager.create_dir(path_kmer2vec, "local")
        file_manager.create_dir(path_read2genome, "local")

        ###################################
        # --------- Training model --------#
        ###################################

        # If tax_taken is given, need to change the file

        if tax_taken is not None:
            data, labels = data_manager.filter_fastdna_fasta_file(
                data, labels, path_tmp_folder, tax_taken
            )

        subprocess.call(
            [
                os.path.join(path_fastDNA, "fastdna"),
                "supervised",
                "-input",
                data,
                "-labels",
                labels,
                "-output",
                path_model,
                "-minn",
                str(k_mer_size),
                "-maxn",
                str(k_mer_size),
                "-dim",
                str(embedding_size),
                "-epoch",
                str(n_steps),
                "-lr",
                str(learning_rate),
                "-thread",
                str(n_cpus),
                "-noise",
                str(noise),
                "-length",
                str(max_length),
            ]
        )

        if tax_taken is not None:
            subprocess.call(["rm", data])
            subprocess.call(["rm", labels])

        ###################################
        # ---------- Saving files ---------#
        ###################################

        df = pd.read_csv(path_model + ".vec", sep=" ", header=None, skiprows=1)
        reverse_index = df[0].to_dict()
        dico_index = {}
        for _, v in reverse_index.items():
            dico_index[str(v)] = int(k_mer_size)

        final_embeddings = np.array(df[np.arange(1, df.shape[1])])
        data_manager.save_embeddings(
            path_embedding, final_embeddings, dico_index, reverse_index, path_tmp_folder
        )
