import numpy as np
import pandas as pd
import os
import subprocess
import logging
from metagenome2vec.utils import parser_creator, data_manager, file_manager

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
    logging.basicConfig(filename=os.path.join(args.path_log, args.log_file), level=logging.DEBUG)


    ###################################
    #-------- Learning fastext -------#
    ###################################

    path_save = os.path.join(path_analysis, "kmer2vec", "genome", catalog, file_model_name, parameter_structu, param_fastext)

    # Create the folder where the model will be saved
    file_manager.create_dir(path_save, "local")

    path_fasttext = os.getenv("FASTTEXT")
    assert path_fasttext is not None, "FASTTEXT environment variable has to be defined"
    logging.info("Start FastText Training")
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
    
    logging.info("Saving files")
    df = pd.read_csv(os.path.join(path_save, "vectors.vec"), sep=" ", header=None, skiprows=1)
    reverse_index = df[0].to_dict()
    dico_index = {}
    for k, v in reverse_index.items():
        dico_index[str(v)] = int(k)

    final_embeddings = np.array(df[np.arange(1, df.shape[1])])
    data_manager.save_embeddings(path_save, final_embeddings, dico_index, reverse_index, path_tmp_folder)
    subprocess.call(['rm', os.path.join(path_save, "vectors.vec")])

