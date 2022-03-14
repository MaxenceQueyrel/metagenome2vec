import numpy as np
import pandas as pd
import subprocess
import os

from metagenome2vec.utils import file_manager, parser_creator, data_manager

if __name__ == "__main__":

    ###################################
    #------ Script's Parameters ------#
    ###################################

    args = parser_creator.ParserCreator().parser_fastdna()
    data, labels = args.path_data.split(",")
    f_name = args.f_name
    tax_taken = None if (args.tax_taken is None or args.tax_taken == "None") else [str(x) for x in args.tax_taken.split('.')]

    if tax_taken is not None:
        f_name += "_n_tax_%s" % len(tax_taken)
    path_save = os.path.join(args.path_kmer2vec, f_name)
    output = os.path.join(args.path_model, f_name)

    path_fastDNA = os.getenv("FASTDNA")
    assert path_fastDNA is not None, "FASTDNA environment variable has to be defined"

    ###################################
    # ---------- Create folders -------#
    ###################################

    # Create the folder where the model will be saved
    file_manager.create_dir(path_save, "local")
    file_manager.create_dir(args.path_model, "local")

    ###################################
    #--------- Training model --------#
    ###################################

    # If tax_taken is given, need to change the file

    if tax_taken is not None:
        data, labels = data_manager.filter_fastdna_fasta_file(data, labels, args.path_tmp_folder, tax_taken)

    subprocess.call([os.path.join(path_fastDNA, "fastdna"),
                     "supervised",
                     "-input", data,
                     "-labels", labels,
                     "-output", output,
                     "-minn", str(args.k_mer_size),
                     "-maxn", str(args.k_mer_size),
                     "-dim", str(args.embedding_size),
                     "-epoch", str(args.n_steps),
                     "-lr", str(args.learning_rate),
                     "-thread", str(args.n_cpus),
                     "-noise", str(args.noise),
                     "-length", str(args.max_length)])

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
        dico_index[str(v)] = int(args.k_mer_size)

    final_embeddings = np.array(df[np.arange(1, df.shape[1])])
    data_manager.save_embeddings(path_save, final_embeddings, dico_index, reverse_index, args.path_tmp_folder)
