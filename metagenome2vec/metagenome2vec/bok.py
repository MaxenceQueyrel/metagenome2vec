import os
import sys
import numpy as np
import pandas as pd
import logging

from metagenome2vec.utils import transformation_ADN
from metagenome2vec.utils.string_names import *


def transform(spark, k, path_data, df_metadata, overwrite):

    path_save = os.path.join(path_data, "bok.csv")

    #metadata = pd.read_csv(path_metadata, delimiter=",")[[id_fasta_name, group_name]]
    if not overwrite and os.path.isfile(path_save):
        logging.info("File already exists !")
        sys.exit(0)
    logging.info("Begin BoK structuring")
    X = []
    # idx_to_del corresponds to kmer complement
    idx_to_del = [i for i in range(4 ** k) if transformation_ADN.int_to_kmer(i, k)[0] in ["G", "T"]]
    m = 0
    for metagenome in df_metadata[id_fasta_name]:
        if not os.path.exists(os.path.join(path_data, metagenome)):
            logging.info("Interation %s, Metagenome : %s doesn't exist" % (m, metagenome))
            continue
        logging.info("Interation %s, Metagenome : %s" % (m, metagenome))
        kmer_count = spark.read.parquet(os.path.join(path_data, metagenome)).select(kmer_name, "count")
        kmer_count = kmer_count.filter(~kmer_count.kmer.rlike(r'[^ACGT]'))
        kmer_count = kmer_count.rdd.collectAsMap()
        X.append([0] * (4 ** k))
        for kmer, count in kmer_count.items():
            idx = transformation_ADN.kmer_to_int(str(kmer))
            X[m][idx] = count
        X[m] = [x for i, x in enumerate(X[m]) if i not in idx_to_del]
        m += 1
    X = np.array(X)
    columns = [transformation_ADN.int_to_kmer(x, k) for x in range(4 ** k)]
    columns = [col for i, col in enumerate(columns) if i not in idx_to_del]
    X = pd.DataFrame(X, columns=columns)
    X = pd.concat([df_metadata, X], axis=1)
    X.to_csv(path_save, index=False, sep=",")
    logging.info("End computation")