# -*- coding: utf-8 -*-

import sys
import os
import subprocess
import numpy as np
import pandas as pd

root_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "utils"))

import parser_creator
import data_manager
import hdfs_functions as hdfs
import logger


if __name__ == "__main__":

    ###################################
    #------ Script's Parameters ------#
    ###################################

    parser = parser_creator.ParserCreator()
    args = parser.parser_GloVe_genome()

    vocabulary_file_name = args.vocabulary_file_name
    cooccurrence_file_name = args.cooccurrence_file_name
    embedding_size = args.embedding_size
    n_steps = args.n_steps
    step = args.step
    x_max = args.x_max
    learning_rate = args.learning_rate
    window = args.window

    param_glove = "embSize_%s_nStep_%s_xMax_%s_learRate_%s" % (embedding_size, n_steps, x_max, learning_rate)
    file_model_name = "glove"
    file_embed_name = "embeddings.csv"
    path_analysis = args.path_analysis
    path_tmp_folder = args.path_tmp_folder
    path_data = args.path_data
    n_cpus = args.n_cpus
    k_mer_size = args.k_mer_size
    catalog = args.catalog
    parameter_structu = "k_%s_w_%s_s_%s" % (k_mer_size, window, step)

    path_log = args.path_log
    log_file = args.log_file

    log = logger.Logger(path_log, log_file, log_file,
                        variable_time={"k": k_mer_size, "w": window, "s": step,
                                       "kmer2vec": file_model_name},
                        **vars(args))

    ###################################
    #--------- Learning glove --------#
    ###################################

    path_save = os.path.join(path_analysis, "kmer2vec", "genome", catalog, "glove", parameter_structu, param_glove)
    file_word_count = "%s_word_count_glove.txt" % param_glove
    file_cooccurrence = "%s_cooccurrence_glove.bin" % param_glove
    path_GloVe = os.getenv("GLOVE")
    assert path_GloVe is not None, "GLOVE environment variable has to be defined"
    # Create the folder where the model will be saved
    hdfs.create_dir(path_save, "local")
    subprocess.call([os.path.join(os.path.dirname(os.path.realpath(__file__)), "glove.sh"), file_word_count, file_cooccurrence, str(embedding_size),
                     str(n_steps), str(x_max), str(learning_rate), os.path.join(path_save, "tmp"),
                     path_tmp_folder, str(window), path_data, str(n_cpus)])

    ###################################
    #---------- Saving files ---------#
    ###################################

    df = pd.read_csv(os.path.join(path_save, "tmp.txt"), sep=" ", header=None)
    reverse_index = df[0].to_dict()
    dico_index = {}
    for k, v in reverse_index.items():
        dico_index[str(v)] = int(k)

    final_embeddings = np.array(df[np.arange(1, df.shape[1])])
    data_manager.save_embeddings(path_save, final_embeddings, dico_index, reverse_index, path_tmp_folder)
    subprocess.call(["rm", os.path.join(path_save, "tmp.txt"), os.path.join(path_save, "tmp.bin")])

    log.close()
