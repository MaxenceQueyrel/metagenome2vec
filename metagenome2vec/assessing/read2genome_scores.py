import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn
from scipy import stats
from sklearn.metrics import confusion_matrix
from pyspark.sql import Window
import pyspark.sql.functions as F
import csv
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from multiprocessing import Pool
import subprocess
from matplotlib import rcParams

rcParams.update({"figure.autolayout": True})
import os

from metagenome2vec.utils.string_names import *


def get_accuracy(y_true, y_pred):
    try:
        return accuracy_score(y_true, y_pred)
    except:
        return 0


def get_precision(y_true, y_pred):
    try:
        return precision_score(y_true, y_pred, average=None).mean()
    except:
        return 0


def get_recall(y_true, y_pred):
    try:
        return recall_score(y_true, y_pred, average=None).mean()
    except:
        return 0


def get_f1_score(y_true, y_pred):
    try:
        return f1_score(y_true, y_pred, average=None).mean()
    except:
        return 0


def compute_rate_reject_class(df_count_reject, df_count_label, tax_level):
    df_count_reject = df_count_reject.merge(
        df_count_label.rename(columns={tax_level: tax_level + "_2"}),
        how="left",
        left_index=True,
        right_index=True,
    )
    return df_count_reject[tax_level] / df_count_reject[tax_level + "_2"]


def save_bowtie_score(
    path_save,
    df,
    threshold,
    bowtie_index,
    n_cpus,
    path_tmp_folder=None,
    n_reads=-1,
    dataset="train",
    file_mode="w",
):
    df[pred_threshold_name] = df[pred_name].where(df[prob_name] > threshold, other=-1)
    mask = df[pred_threshold_name] != -1
    df_accepted, df_reject = df[mask], df[~mask]
    path_tmp_folder = "./" if path_tmp_folder is None else path_tmp_folder
    n_reads = n_reads if n_reads > 0 and n_reads < df.shape[0] else df.shape[0]
    tmp_file = os.path.join(path_tmp_folder, ".reads.txt")

    def compute_score_bowtie_df(df):
        df.iloc[:n_reads][read_name].to_csv(tmp_file, header=False, index=False)
        res = int(
            subprocess.check_output(
                "%s/bowtie --threads %s -a -v 2 %s --suppress 1,5,6,7 -r %s | grep '\n' | wc -l"
                % (os.getenv("BOWTIE"), bowtie_index, n_cpus, tmp_file),
                shell=True,
            )
            .decode()
            .replace("\n", "")
        )
        subprocess.call(["rm", tmp_file])
        return res / n_reads

    res_accepted = compute_score_bowtie_df(df_accepted)
    res_reject = compute_score_bowtie_df(df_reject)

    with open(path_save, file_mode) as f:
        writer = csv.writer(f)
        if file_mode == "w":
            writer.writerow(
                [
                    "dataset",
                    "n_reads",
                    "threshold",
                    "bowtie_mean_score_accepted",
                    "bowtie_mean_score_reject",
                ]
            )
        writer.writerow([dataset, n_reads, threshold, res_accepted, res_reject])


def compute_metrics(df, threshold, df_count_label, tax_level):
    df[pred_threshold_name] = df[pred_name].where(df[prob_name] > threshold, other=-1)
    mask = df[pred_threshold_name] != -1
    df_accepted, df_reject = df[mask], df[~mask]
    accuracy_threshold = get_accuracy(df_accepted[tax_level], df_accepted[pred_name])
    precision_threshold = get_precision(df_accepted[tax_level], df_accepted[pred_name])
    recall_threshold = get_recall(df[tax_level], df[pred_threshold_name])
    f1_score_threshold = get_f1_score(df[tax_level], df[pred_threshold_name])
    df_count_reject = pd.DataFrame(df_reject[tax_level].value_counts())
    reject_rate = df_reject.shape[0] / df.shape[0]
    reject_class_rate = compute_rate_reject_class(df_count_reject, df_count_label)
    df_reject.shape[0] / df.shape[0]
    return (
        reject_rate,
        accuracy_threshold,
        precision_threshold,
        recall_threshold,
        f1_score_threshold,
        reject_class_rate,
    )


def compute_metrics_mp(args):
    return compute_metrics(*args)


def compute_metrics_by_threshold(df, df_count_label, range_threshold, n_cpus):
    L_reject_rate = []
    L_accuracy = []
    L_precision = []
    L_recall = []
    L_f1_score = []
    D_label_index = {v: k for k, v in enumerate(df_count_label.index)}
    A_rate_reject_class = np.zeros(len(df_count_label))
    pool = Pool(processes=n_cpus)
    res = pool.map(
        compute_metrics_mp,
        [(df, threshold, df_count_label) for threshold in range_threshold],
    )
    for rej_rate, acc, prec, rec, f1, rate_rej_class in res:
        L_accuracy.append(acc)
        L_reject_rate.append(rej_rate)
        L_precision.append(prec)
        L_recall.append(rec)
        L_f1_score.append(f1)
        for i, r in rate_rej_class.iteritems():
            A_rate_reject_class[D_label_index[i]] += r
    return (
        np.array(L_reject_rate),
        np.array(L_accuracy),
        np.array(L_precision),
        np.array(L_recall),
        np.array(L_f1_score),
        A_rate_reject_class,
    )


def heatmap(y_true, y_pred, D_id, path_save, dataset, title):
    for normallize in ["true", "all"]:
        path_save_fig = os.path.join(
            path_save, "heatmap_%s_norm_%s.png" % (dataset, normallize)
        )
        plt.figure(figsize=(15, 15))
        df = confusion_matrix(y_true, y_pred, labels=list(map(int, D_id.keys())))
        df = df * 1.0 / df.sum(axis=1).reshape(len(df), 1)
        df = np.where(np.isnan(df), 0, df)
        m = df.max()
        if m == 0:
            m = 1
        labels = D_id.values()
        ax = seaborn.heatmap(
            df,
            cmap="YlGnBu",
            xticklabels=labels,
            yticklabels=labels,
            annot=False,
            vmin=0,
            vmax=m,
        )
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_yticklabels(labels, fontsize=10)
        cbar = ax.collections[0].colorbar
        cbar.set_ticklabels(
            [str(round(x, 1)) + "%" for x in np.arange(0, m + m / 5.0, m / 5.0)]
        )
        ax.set_xlabel("True labels", fontsize=14)
        ax.set_ylabel("Predictions", fontsize=14)
        ax.set_title(title, fontsize=14)
        plt.savefig(path_save_fig)
        plt.tight_layout()
        plt.close()


def plot_reject_by_abundance(path_save, df_count_label, A_count_reject, D_id, title):
    index = df_count_label.index
    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot(111)
    labels = [D_id[str(idx)] for idx in index]
    ax.set_xticklabels(labels, rotation="vertical")
    ax.bar(range(len(A_count_reject)), A_count_reject)
    ax.set_xlabel("Index for the species id")
    ax.set_ylabel("Rejected reads rate")
    ax.set_title(title)
    fig.savefig(path_save)
    plt.close()


def plot_proportion_true_pred(path_save, df_proportion, sim_id, title):
    df_proportion = df_proportion[df_proportion[sim_id_name] == sim_id]
    plt.figure(figsize=(8, 8))
    ax = seaborn.regplot(
        x=prop_true_name,
        y=prop_pred_name,
        data=df_proportion,
        scatter=True,
        fit_reg=True,
    )
    ax.set_xlabel("True abundance proportion")
    ax.set_ylabel("Predicted abundance proportion")
    ax.set_title(title)
    plt.savefig(path_save)
    plt.close()


def save_score(
    path_save, df, df_count_label, threshold=0.0, dataset="train", file_mode="w"
):
    df.to_csv(os.path.join(path_save, "res_prob_%s.csv" % dataset), index=False)
    with open(os.path.join(path_save, "metrics.csv"), file_mode) as f:
        writer = csv.writer(f)
        nb_class = len(df_count_label)
        _, accuracy, precision, recall, F1_score, _ = compute_metrics(
            df, threshold, df_count_label
        )
        if file_mode == "w":
            writer.writerow(
                [
                    "dataset",
                    "nb_class",
                    "threshold",
                    "accuracy",
                    "precision",
                    "recall",
                    "F1_score",
                ]
            )
        writer.writerow(
            [dataset, nb_class, threshold, accuracy, precision, recall, F1_score]
        )


def save_cv_metrics(path_save, model):
    try:
        df_scores = model.cross_validation_metrics_summary().as_data_frame()
        df_scores = df_scores.set_index("").astype(float)
        df_scores.to_csv(path_save)
    except:
        return


def create_correlation_table(path_save, df_proportion):
    L_res = []
    for name_sim in df_proportion[sim_id_name].values:
        df_tmp = df_proportion[df_proportion[sim_id_name] == name_sim]
        correlation = round(
            stats.spearmanr(df_tmp[prop_true_name], df_tmp[prop_pred_name])[0], 2
        )
        L_res.append([name_sim, correlation])
    df = pd.DataFrame(L_res, columns=[sim_id_name, "Spearman Correlation"])
    df.to_csv(path_save, index=False)


def plot_score(
    path_save,
    range_threshold,
    A_reject_rate,
    A_accuracy,
    A_precision,
    A_recall,
    A_f1_score,
    title,
):
    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(111)
    ax.plot(range_threshold, A_reject_rate, label="Rejected rate")
    ax.plot(range_threshold, A_accuracy, label="Accuracy")
    ax.plot(range_threshold, A_precision, label="Precision")
    ax.plot(range_threshold, A_recall, label="Recall")
    ax.plot(range_threshold, A_f1_score, label="F1 score")
    ax.legend()
    ax.set_xlabel("reject threshold")
    ax.set_ylabel("score")
    ax.set_title(title)
    fig.savefig(path_save)


def plot_score2(
    path_save, A_reject_rate, A_accuracy, A_precision, A_recall, A_f1_score, title
):
    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(111)
    ax.plot(A_reject_rate, A_accuracy, label="Accuracy")
    ax.plot(A_reject_rate, A_precision, label="Precision")
    ax.plot(A_reject_rate, A_recall, label="Recall")
    ax.plot(A_reject_rate, A_f1_score, label="F1 score")
    ax.legend()
    ax.set_xlabel("reject rate")
    ax.set_ylabel("score")
    ax.set_title(title)
    fig.savefig(path_save)

