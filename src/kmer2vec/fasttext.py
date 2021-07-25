import numpy as np
import pandas as pd
import sys
import os
import subprocess

root_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, os.path.join(root_folder, "utils"))

import parser_creator
import data_manager
import hdfs_functions as hdfs
import logger

if __name__ == "__main__":

    ###################################
    #------ Script's Parameters ------#
    ###################################

    parser = parser_creator.ParserCreator()
    args = parser.parser_fasttext_genome()

    embedding_size = args.embedding_size
    learning_rate = args.learning_rate
    window = args.window
    n_steps = args.n_steps
    step = args.step
    min_ngram = args.min_ngram
    max_ngram = args.max_ngram

    param_fastext = "embSize_%s_nStep_%s_learRate_%s" % (embedding_size, n_steps, learning_rate)
    file_model_name = "fasttext"
    path_data = args.path_data
    path_analysis = args.path_analysis
    n_cpus = args.n_cpus
    k_mer_size = args.k_mer_size
    catalog = args.catalog
    path_log = args.path_log
    path_tmp_folder = args.path_tmp_folder
    log_file = args.log_file
    parameter_structu = "k_%s_w_%s_s_%s" % (k_mer_size, window, step)

    log = logger.Logger(path_log, log_file, log_file,
                        variable_time={"k": k_mer_size, "w": window, "s": step,
                                       "kmer2vec": file_model_name},
                        **vars(args))

    ###################################
    #-------- Learning fastext -------#
    ###################################

    path_save = os.path.join(path_analysis, "kmer2vec", "genome", catalog, file_model_name, parameter_structu, param_fastext)

    # Create the folder where the model will be saved
    hdfs.create_dir(path_save, "local")

    path_fasttext = os.getenv("FASTTEXT")
    assert path_fasttext is not None, "FASTTEXT environment variable has to be defined"
    subprocess.call([os.path.join(path_fasttext, "fasttext"), "skipgram",
                     "-input", path_data,
                     "-output", os.path.join(path_save, "vectors"),
                     "-lr", str(learning_rate),
                     "-dim", str(embedding_size),
                     "-ws", str(window),
                     "-epoch", str(n_steps),
                     "-minCount", str(5),
                     "-neg", str(5),
                     "-loss", "hs",
                     "-bucket", str(2000000),
                     "-minn", str(min_ngram),
                     "-maxn", str(max_ngram),
                     "-thread", str(n_cpus),
                     "-t", str(1e-4),
                     "-lrUpdateRate", str(100)])

    ###################################
    #---------- Saving files ---------#
    ###################################

    df = pd.read_csv(os.path.join(path_save, "vectors.vec"), sep=" ", header=None, skiprows=1)
    reverse_index = df[0].to_dict()
    dico_index = {}
    for k, v in reverse_index.items():
        dico_index[str(v)] = int(k)

    final_embeddings = np.array(df[np.arange(1, df.shape[1])])
    data_manager.save_embeddings(path_save, final_embeddings, dico_index, reverse_index, path_tmp_folder)
    subprocess.call(['rm', os.path.join(path_save, "vectors.vec")])

    log.close()

