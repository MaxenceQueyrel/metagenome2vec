import pandas as pd
import os
import json
import csv
import numpy as np
import subprocess
import dill as pickle
import re
from tqdm import tqdm

from metagenome2vec.utils.stat_func import ttest_cv
from metagenome2vec.utils.string_names import *


def load_embeddings(path_embeddings, skip_kmer_name=True, L_kmer_to_del=None):
    """
    Load the embeddings and the corresponding directories

    :param path_embeddings: String, path to embeddings
    :param skip_kmer_name, Boolean, default=True, don't read the first column of kmer (usefull for torchtext and others)
    :param L_kmer_to_del, List, default=["<unk>", "</s>"] kmers we want to del
    :return: embeddings, Numpy N D array [size voc, size embed]
             dico_index, dict key=kmer (string), value=index (int) in embeddings
             reverse_dico, dict key=index (int) in embedings, value=kmer (string)
    """
    with open(os.path.join(path_embeddings, "reverse_index.json"), "r") as fp:
        reverse_index = json.load(
            fp, object_hook=lambda d: {int(k): str(v) for k, v in list(d.items())}
        )
    with open(os.path.join(path_embeddings, "index.json"), "r") as fp:
        dico_index = json.load(
            fp, object_hook=lambda d: {str(k): int(v) for k, v in list(d.items())}
        )
    embeddings = pd.read_csv(
        os.path.join(path_embeddings, "embeddings.csv"), sep=" ", header=None
    )
    if L_kmer_to_del is not None:
        L_index_to_del = [dico_index[x] for x in L_kmer_to_del if x in dico_index]
        for kmer in L_kmer_to_del:
            if kmer in dico_index:
                dico_index.pop(kmer)
        embeddings = embeddings.drop(L_index_to_del).reset_index(drop=True)
        reverse_index = embeddings[0].to_dict()
        dico_index = {v: k for k, v in reverse_index.items()}
    if skip_kmer_name:
        embeddings = np.array(embeddings.values[:, 1:], dtype=float)
    else:
        embeddings = embeddings.values
    return embeddings, dico_index, reverse_index


def load_fasttext(path_embeddings):
    """
    Load the embeddings and the corresponding directories

    :param path_embeddings: String, path to embeddings
    :return: path_model, Path to the fasttext model
             dico_index, dict key=kmer (string), value=index (int) in embeddings
             reverse_dico, dict key=index (int) in embedings, value=kmer (string)
    """
    with open(os.path.join(path_embeddings, "reverse_index.json"), "r") as fp:
        reverse_index = json.load(
            fp, object_hook=lambda d: {int(k): str(v) for k, v in list(d.items())}
        )
    with open(os.path.join(path_embeddings, "index.json"), "r") as fp:
        dico_index = json.load(
            fp, object_hook=lambda d: {str(k): int(v) for k, v in list(d.items())}
        )
    return os.path.join(path_embeddings, "vectors.bin"), dico_index, reverse_index


def save_embeddings(
    path_embeddings, embeddings, dico_index, reverse_index=None, path_tmp_folder="./"
):
    """
    Save the embeddings, index (if given) and reverse_index if given
    :param path_embeddings: path where are saved the embeddings
    :param embeddings: Numpy 2D array, contains the embeddings for each k-mers
    :param dico_index: dictionary, key=kmer (string), value=index (int) in embeddings
    :param reverse_index: dictionary, key=index (int) in embedings, value=kmer (string)
    :param path_tmp_folder: Str (default './'), path to the tmp folder
    :return:
    """
    #
    file_embed_name = "embeddings.csv"
    file_embed_name_tmp = os.path.join(path_tmp_folder, ".embeddings_tmp.csv")
    tmp_file = os.path.join(path_tmp_folder, ".tmp_file")
    if np.isnan(embeddings[0, -1]):
        embeddings = embeddings[:, :-1]
    np.savetxt(file_embed_name_tmp, embeddings)
    l = []
    for i in range(len(dico_index)):
        l.append(reverse_index[i])
    with open(tmp_file, "w") as f:
        for item in l:
            f.write("%s\n" % item)
    subprocess.call(
        "paste -d' ' %s %s > %s ; rm %s ; rm %s"
        % (
            tmp_file,
            file_embed_name_tmp,
            os.path.join(path_embeddings, file_embed_name),
            tmp_file,
            file_embed_name_tmp,
        ),
        shell=True,
    )

    with open(
        os.path.join(
            path_embeddings,
            file_embed_name.replace(".csv", ".json").replace("embeddings", "index"),
        ),
        "w",
    ) as fp:
        json.dump(dico_index, fp)
    if reverse_index is not None:
        with open(
            os.path.join(
                path_embeddings,
                file_embed_name.replace(".csv", ".json").replace(
                    "embeddings", "reverse_index"
                ),
            ),
            "w",
        ) as fp:
            json.dump(reverse_index, fp)


def balance_data(df, name_col, n_sample_by_class=1000, mode="both"):
    """
    Resample a dataset with n_sample_by_class elements for each class
    :param df: Pandas dataframe
    :param name_col: str, name of the column to balance
    :param n_sample_by_class: int
    :param mode: str, "both" (for under and over sampling), "under" or "over"
    :return: df_res: Pandas dataframe
    """
    counts = df[name_col].value_counts()
    df_res = None
    for ind, count in tqdm(counts.items()):
        if count < n_sample_by_class:
            if mode in ["both", "over"]:
                to_add = n_sample_by_class - count
                df_keep = df[df[name_col] == ind]
                df_tmp = df_keep.sample(to_add, replace=True)
                df_tmp = pd.concat([df_keep, df_tmp], axis=0)
            else:
                df_tmp = df[df[name_col] == ind]
        else:
            if mode in ["both", "under"]:
                df_tmp = df[df[name_col] == ind].sample(
                    n_sample_by_class, replace=False
                )
            else:
                df_tmp = df[df[name_col] == ind]
        if df_res is None:
            df_res = df_tmp
        else:
            df_res = pd.concat([df_res, df_tmp], axis=0)
    return df_res


def load_simulated_data(path_data, name_simulated_data, name_matrix_save):
    """
    Load or create the simulation data matrix
    :param path_data: String, Path to folder, assuming that it exists a file named reads and a file
    named mapping_read_genome
    :param name_simulated_data: str, name of the simulation data loaded
    :param name_matrix_save: str, name of the matrix that is saved
    :return: Matrix, numpy 2D array, first column is read second is class
    """
    path_final_matrix = os.path.join(path_data, name_matrix_save)
    if not os.path.exists(path_final_matrix):
        path_mapping_read_genome = os.path.join(path_data, "mapping_read_genome")
        path_reads = os.path.join(path_data, name_simulated_data)
        mapping_read_genome = pd.read_csv(path_mapping_read_genome, sep="\t", dtype=str)
        reads = pd.read_csv(path_reads, sep="\t")  # 1 min
        read_id_name = "read_id"
        tax_id_name = "tax_id"
        sim_id_name = "sim_id"
        read_name = "read"
        mapping_read_genome = mapping_read_genome[[tax_id_name, read_id_name]]
        reads[read_id_name] = reads[read_id_name].apply(
            lambda x: re.sub("(.*)-.*$", "\\1", x.replace("@", ""))
        )
        reads = reads.merge(mapping_read_genome, on=read_id_name)[
            [read_name, tax_id_name, sim_id_name]
        ]
        sim_ids = reads[sim_id_name].unique()
        reads["ratio_genome_by_sim"] = np.zeros(reads.shape[0], dtype=np.float)
        # reads["train_valid"] = np.zeros(reads.shape[0], dtype=np.str)
        # train_sim_ids, valid_sim_ids = train_test_split(sim_ids, test_size=1-train_ratio)
        for sim_id in tqdm(sim_ids):
            tmp = reads[reads[sim_id_name] == sim_id]
            tmp = (
                tmp.groupby(tax_id_name).transform("count")["ratio_genome_by_sim"]
                * 1.0
                / tmp.shape[0]
            )
            reads.loc[tmp.index, "ratio_genome_by_sim"] = tmp
            # reads.loc[tmp.index, "train_valid"] = "train" if sim_id in train_sim_ids else "valid"
        # reads_train = reads[reads["train_valid"] == "train"]
        # reads_valid = reads[reads["train_valid"] == "valid"]
        # Removing potential read that are in the train set and the validation set
        # reads_train = reads_train[~reads_train[read_name].isin(reads_valid[read_name])]
        # reads = pd.concat([reads_train, reads_valid])
        reads.to_csv(path_final_matrix, index=False, sep="\t")
        return reads
    else:
        return pd.read_csv(path_final_matrix, sep="\t")


def process_score_cv(score_cv):
    score_cv_str = "["
    for i in range(len(score_cv)):
        if i != len(score_cv) - 1:
            score_cv_str += "%.3f," % round(score_cv[i], 3)
        else:
            score_cv_str += "%.3f]" % round(score_cv[i], 3)
    std_score = np.std(score_cv)
    return score_cv_str, np.round(std_score, 3)


def write_file_res_benchmarck_classif(path_file, matrix_name, classifier, scores):
    """
    Write into the result file of the benchmark classif
    :param path_file: String, complete path to the file res
    :param matrix_name: String, name of the matrix used for learning
    :param classifier: String, name of the classifier used for learning
    :param scores: Dictionary, all scores computed by the benchmark
    :return:
    """
    read_option = "a" if os.path.isfile(path_file) else "w"
    with open(path_file, read_option, newline="") as f_res:
        L_metric = [
            "matrix_name",
            "classifier",
            "fit_time",
            "score_time",
            "test_accuracy",
            "train_accuracy",
            "test_f1",
            "train_f1",
            "test_precision",
            "train_precision",
            "test_recall",
            "train_recall",
            "test_roc_auc",
            "train_roc_auc",
            "test_accuracy_cv",
            "std_test_accuracy",
            "test_precision_cv",
            "std_test_precision",
            "test_f1_cv",
            "std_test_f1",
            "test_recall_cv",
            "std_test_recall",
            "test_roc_auc_cv",
            "std_test_roc_auc",
            "significant",
        ]
        f_res_csv = csv.writer(f_res, delimiter=";", quotechar="|")
        if read_option == "w":
            f_res_csv.writerow(L_metric)
        test_accuracy_cv_str, std_test_accuracy = process_score_cv(
            scores["test_accuracy"]
        )
        test_precision_cv_str, std_test_precision = process_score_cv(
            scores["test_precision"]
        )
        test_f1_cv_str, std_test_f1 = process_score_cv(scores["test_f1"])
        test_recall_cv_str, std_test_recall = process_score_cv(scores["test_recall"])
        test_roc_auc_cv_str, std_test_roc_auc = process_score_cv(scores["test_roc_auc"])
        L_metric = [metric for metric in L_metric if metric in scores]
        f_res_csv.writerow(
            [matrix_name, classifier]
            + ["{0:.3f}".format(np.mean(scores[metric])) for metric in L_metric]
            + [
                test_accuracy_cv_str,
                std_test_accuracy,
                test_precision_cv_str,
                std_test_precision,
                test_f1_cv_str,
                std_test_f1,
                test_recall_cv_str,
                std_test_recall,
                test_roc_auc_cv_str,
                std_test_roc_auc,
            ]
            + [""]
        )

    # Compute significant test for the best model against the other
    res = pd.read_csv(path_file, sep=";")
    res = res.sort_values("test_accuracy", ascending=False)
    best_scores = res.iloc[0]
    best_index = best_scores.name
    accuracy_cv_best = eval(best_scores["test_accuracy_cv"])
    for idx, row in res.iterrows():
        if best_index == idx:
            res.at[idx, "significant"] = 0.0
        else:
            accuracy_cv_candidate = eval(res["test_accuracy_cv"].loc[idx])
            s, p = ttest_cv(accuracy_cv_best, accuracy_cv_candidate)
            res.at[idx, "significant"] = "%.3f" % round(p, 3)
    res.to_csv(path_file, sep=";", index=False, float_format="%.3f")


def open_file_res_read_classif(path_file):
    """
    open a writable file for the benchmark classif
    :param path_file: String, complete path to the file res
    :return: FILE
    """
    if os.path.isfile(path_file):
        f_res = open(path_file, "a")
    else:
        f_res = open(path_file, "w")
        f_res.write(
            "read_model;classifier;nb_class;best_accuracy_gs_cv;accuracy_cv;mean_accuracy_cv;mean_accuracy_cv_train\n"
        )
    return f_res


def open_file_res_read2genome(path_file):
    """
    open a writable file for read2genome scores
    :param path_file: String, complete path to the file res
    :return: FILE
    """
    if os.path.isfile(path_file):
        f_res = open(path_file, "a")
    else:
        f_res = open(path_file, "w")
        f_res.write(
            "read2genome;nb_class;threshold;accuracy(train,val);precision(train,val);recall(train,val);F1_score(train,val)\n"
        )
    return f_res


def save_sklearn_model(model, path_model, overwrite=False):
    if not os.path.exists(path_model) or overwrite is True:
        pickle.dump(model, open(path_model, "wb"))


def load_sklearn_model(path_model):
    # load the model from disk
    return pickle.load(open(path_model, "rb"))


def load_df_taxonomy_ref(path_tax_report):
    """
    load the dataframe associated with the taxonomy information
    :param path_tax_report: String, path to the csv file containing the taxonomy information
    :return:
    """
    df_taxonomy_ref = pd.read_csv(path_tax_report)
    df_taxonomy_ref[ncbi_id_name] = df_taxonomy_ref[ncbi_id_name].apply(int)
    df_taxonomy_ref = df_taxonomy_ref.applymap(str)
    return df_taxonomy_ref


def filter_fastdna_fasta_file(data, labels, path_tmp_folder, tax_taken):
    tmp_file_reads = os.path.join(
        path_tmp_folder, data.split("/")[-1] + "_%s" % len(tax_taken)
    )
    tmp_file_tax = os.path.join(
        path_tmp_folder, labels.split("/")[-1] + "_%s" % len(tax_taken)
    )
    with open(tmp_file_reads, "w") as reads_out:
        with open(tmp_file_tax, "w") as tax_out:
            with open(data, "r") as reads_in:
                with open(labels, "r") as tax_in:
                    for i, read in enumerate(reads_in):
                        if i % 2 == 1:
                            tax = tax_in.readline().replace("\n", "")
                            if tax in tax_taken:
                                read = read.replace("\n", "")
                                reads_out.write(">%s\n" % (i // 2))
                                reads_out.write("%s\n" % read)
                                tax_out.write("%s\n" % tax)
    return tmp_file_reads, tmp_file_tax
