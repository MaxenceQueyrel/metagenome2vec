# -*- coding: utf-8 -*-

import numpy as np
import sys
import os
from pyspark.ml.feature import Word2Vec
from pyspark.sql.functions import split
import subprocess
import json

root_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, os.path.join(root_folder, "utils"))
import hdfs_functions as hdfs
import parser_creator
import logger
import data_manager


if __name__ == "__main__":

    ###################################
    # ------ Script's Parameters ------#
    ###################################
    parser = parser_creator.ParserCreator()
    args = parser.parser_word2vec_genome()

    # Neural Network parameters
    learning_rate = args.learning_rate
    n_steps = args.n_steps
    step = args.step
    embedding_size = args.embedding_size
    window = args.window
    n_cpus = args.n_cpus
    minCount = 5
    spark_conf = args.spark_conf

    param_word2vec = "embSize_%s_nStep_%s_learRate_%s/" % (
        embedding_size,
        n_steps,
        learning_rate,
    )
    file_embed_name = "embeddings.csv"
    file_model_name = "glove"
    path_data = args.path_data
    path_analysis = args.path_analysis
    k_mer_size = args.k_mer_size
    catalog = args.catalog
    parameter_structu = "k_%s_w_%s_s_%s" % (k_mer_size, window, step)
    path_tmp_folder = args.path_tmp_folder
    path_log = args.path_log
    log_file = args.log_file

    log = logger.Logger(
        path_log,
        log_file,
        log_file,
        variable_time={
            "k": k_mer_size,
            "w": window,
            "s": step,
            "kmer2vec": file_model_name,
        },
        **vars(args)
    )

    ###################################
    # ---------- Spark Conf -----------#
    ###################################

    spark = hdfs.createSparkSession("kmer2vec", **spark_conf)

    ###################################
    # ------------- Model -------------#
    ###################################

    data = spark.read.text(path_data)
    split_col = split(data["value"], " ")
    data = data.withColumn("value", split_col)
    data = data.persist()
    data.count()

    word2vec = Word2Vec(inputCol="value", outputCol="result", seed=42)
    word2vec.setMinCount(minCount).setMaxIter(n_steps).setVectorSize(
        embedding_size
    ).setWindowSize(window).setStepSize(learning_rate).setNumPartitions(n_cpus)

    model = word2vec.fit(data)

    ###################################
    # ---------- Saving files ---------#
    ###################################

    path_save = os.path.join(
        path_analysis,
        "kmer2vec",
        "genome",
        catalog,
        "word2vec",
        parameter_structu,
        param_word2vec,
    )
    hdfs.create_dir(path_save, "local")

    vectors = model.getVectors().toPandas()
    reverse_index = vectors["word"].to_dict()
    dico_index = {}
    for k, v in reverse_index.items():
        dico_index[str(v)] = int(k)

    final_embeddings = np.array([x.toArray() for x in list(vectors["vector"])])
    data_manager.save_embeddings(
        path_save, final_embeddings, dico_index, reverse_index, path_tmp_folder
    )
    log.close()
