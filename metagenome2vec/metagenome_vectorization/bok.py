# -*- coding: latin-1 -*-

import os
import sys
import numpy as np
import pandas as pd
root_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, os.path.join(root_folder, "utils"))
if sys.version_info[0] == 3 and sys.version_info[1] == 7:
    import transformation_ADN
else:
    import transformation_ADN2 as transformation_ADN
import logger
import parser_creator
import hdfs_functions as hdfs
from string_names import *


if __name__ == "__main__":

    ####################################
    # ------ Script's Parameters ------#
    ####################################

    parser = parser_creator.ParserCreator()
    args = parser.parser_BoK()
    k = args.k_mer_size
    mode = args.mode
    path_data = args.path_data
    path_save = os.path.join(path_data, "bok.csv")
    log_file = args.log_file
    overwrite = args.overwrite
    path_log = args.path_log
    nb_metagenome = args.nb_metagenome
    path_metadata = args.path_metadata

    log = logger.Logger(path_log, log_file, log_file,
                        variable_time={"k": k},
                        **vars(args))

    ####################################
    # ---------- Spark Conf -----------#
    ####################################

    spark = hdfs.createSparkSession("BoK")

    #############################
    #------ Computing BoK ----- #
    #############################

    metadata = pd.read_csv(path_metadata, delimiter=",")[[id_fasta_name, group_name]]
    if not overwrite and os.path.isfile(path_save):
        log.write("File already exists !")
        sys.exit(0)
    log.write("Begin BoK structuring")
    X = []
    # idx_to_del corresponds to kmer complement
    idx_to_del = [i for i in range(4 ** k) if transformation_ADN.int_to_kmer(i, k)[0] in ["G", "T"]]
    m = 0
    for metagenome in metadata[id_fasta_name]:
        if not os.path.exists(os.path.join(path_data, metagenome)):
            log.write("Interation %s, Metagenome : %s doesn't exist" % (m, metagenome))
            continue
        log.write("Interation %s, Metagenome : %s" % (m, metagenome))
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
    X = pd.concat([metadata, X], axis=1)
    X.to_csv(path_save, index=False, sep=",")
    log.write("End computation")

