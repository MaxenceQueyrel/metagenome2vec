import numpy as np
import pandas as pd
import subprocess
import sys
import os

root_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, os.path.join(root_folder, "utils"))

import parser_creator
import data_manager
import hdfs_functions as hdfs


if __name__ == "__main__":

    ###################################
    #------ Script's Parameters ------#
    ###################################

    args = parser_creator.ParserCreator().parser_fastdna()

    k = args.k_mer_size
    dim = args.embedding_size
    epoch = args.n_steps
    lr = args.learning_rate
    data, labels = args.path_data.split(",")
    thread = args.n_cpus
    path_model = args.path_model
    f_name = args.f_name
    noise = args.noise
    max_length = args.max_length
    path_kmer2vec = args.path_kmer2vec
    tax_taken = None if (args.tax_taken is None or args.tax_taken == "None") else [str(x) for x in args.tax_taken.split('.')]
    path_tmp_folder = args.path_tmp_folder

    if tax_taken is not None:
        f_name += "_n_tax_%s" % len(tax_taken)
    path_save = os.path.join(path_kmer2vec, f_name)
    output = os.path.join(path_model, f_name)

    path_fastDNA = os.getenv("FASTDNA")
    assert path_fastDNA is not None, "FASTDNA environment variable has to be defined"

    ###################################
    # ---------- Create folders -------#
    ###################################

    # Create the folder where the model will be saved
    hdfs.create_dir(path_save, "local")
    hdfs.create_dir(path_model, "local")

    ###################################
    #--------- Training model --------#
    ###################################

    # If tax_taken is given, need to change the file

    if tax_taken is not None:
        data, labels = data_manager.filter_fastdna_fasta_file(data, labels, path_tmp_folder, tax_taken)

    subprocess.call([os.path.join(path_fastDNA, "fastdna"),
                     "supervised",
                     "-input", data,
                     "-labels", labels,
                     "-output", output,
                     "-minn", str(k),
                     "-maxn", str(k),
                     "-dim", str(dim),
                     "-epoch", str(epoch),
                     "-lr", str(lr),
                     "-thread", str(thread),
                     "-noise", str(noise),
                     "-length", str(max_length)])

    if tax_taken is not None:
        subprocess.call(["rm", data])
        subprocess.call(["rm", labels])

    ###################################
    #---------- Saving files ---------#
    ###################################

    df = pd.read_csv(output + ".vec", sep=" ", header=None, skiprows=1)
    reverse_index = df[0].to_dict()
    dico_index = {}
    for k, v in reverse_index.items():
        dico_index[str(v)] = int(k)

    final_embeddings = np.array(df[np.arange(1, df.shape[1])])
    data_manager.save_embeddings(path_save, final_embeddings, dico_index, reverse_index, path_tmp_folder)
    #subprocess.call(['rm', os.path.join(path_save, "vectors.vec")])
