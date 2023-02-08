from pysparkling import H2OContext
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import re
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
import sys

root_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, os.path.join(root_folder, "utils"))
sys.path.insert(0, os.path.join(root_folder, "read2vec"))
sys.path.insert(0, os.path.join(root_folder, "read2genome"))

import logger
import parser_creator
import hdfs_functions as hdfs
import data_manager
from string_names import *


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


def compute_rate_reject_class(df_count_reject, df_count_label):
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


def compute_metrics(df, threshold, df_count_label):
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


def compute_metrics_by_threshold(df, df_count_label, range_threshold):
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


if __name__ == "__main__":
    parser = parser_creator.ParserCreator()
    args = parser.parser_analyse_read2genome()
    mode = args.mode
    assert (
        len(args.path_data.split(",")) == 2
    ), "You have to give one path for training and another one for validation"
    path_data_train, path_data_valid = args.path_data.split(",")
    path_data_train = (
        path_data_train if path_data_train[-1] != "/" else path_data_train[:-1]
    )
    path_data_valid = (
        path_data_valid if path_data_valid[-1] != "/" else path_data_valid[:-1]
    )
    log_file = args.log_file
    path_log = args.path_log
    n_sample_load = args.n_sample_load
    n_instance = args.n_instance
    path_model = args.path_model if args.path_model[-1] != "/" else args.path_model[:-1]
    path_save = re.sub("\..*$", "", path_model)
    path_save = re.sub("model$", "", path_save)
    path_metadata = args.path_metadata
    bowtie_index = args.bowtie_index

    hdfs.create_dir(path_save, mode="local")
    tax_level = args.tax_level
    log = logger.Logger(path_log, log_file, log_file, **vars(args))
    n_cpus = args.n_cpus
    read2genome = args.read2genome
    tax_taken = (
        None
        if (args.tax_taken is None or args.tax_taken == "None")
        else [str(x) for x in args.tax_taken.split(".")]
    )
    path_tmp_folder = args.path_tmp_folder
    spark = hdfs.createSparkSession(
        "read2genome",
        **{
            "spark.driver.memory": "50g",
            "spark.driver.maxResultSize": "10g",
        }
    )
    hc = None
    if read2genome == "h2oModel":
        hc = H2OContext.getOrCreate(spark)
    log.write("Loading read2genome model")
    r2g = data_manager.load_read2genome(
        read2genome, path_model, hc, path_tmp_folder, device="cpu"
    )
    log.write("Loading data")

    if read2genome == "fastDNA" or read2genome == "transformer":
        df_train = spark.createDataFrame(pd.read_csv(path_data_train, sep="\t"))
        df_valid = spark.createDataFrame(pd.read_csv(path_data_valid, sep="\t"))
    else:
        df_train = spark.read.parquet(path_data_train)
        df_valid = spark.read.parquet(path_data_valid)
    if n_sample_load > 0:
        df_train = df_train.limit(n_sample_load)
        df_valid = df_valid.limit(n_sample_load)
    # For h2o write cv scores
    if read2genome == "h2oModel":
        save_cv_metrics(os.path.join(path_save, "cv_metrics.csv"), r2g.model)

    df_taxonomy_ref = pd.read_csv(path_metadata).astype(str)

    col_name = "%s_name" % tax_level
    D_id = {
        key: value
        for key, value in df_taxonomy_ref[[tax_level, col_name]]
        .set_index(tax_level)
        .to_dict()[col_name]
        .items()
    }
    if tax_taken is not None:
        D_id = {
            key: value
            for key, value in df_taxonomy_ref[[tax_level, col_name]]
            .set_index(tax_level)
            .to_dict()[col_name]
            .items()
            if key in tax_taken
        }
        # take the tax_id rather than the tax level in order to make a filter
        tax_taken = df_taxonomy_ref[df_taxonomy_ref[tax_level].isin(tax_taken)][
            ncbi_id_name
        ].tolist()
        df_train = df_train.filter(
            df_train.tax_id.isin([int(x) for x in tax_taken])
        ).persist()
        df_train.count()
        df_valid = df_valid.filter(
            df_valid.tax_id.isin([int(x) for x in tax_taken])
        ).persist()
        df_valid.count()

    log.write("Prediction on train")
    df_train_pd = r2g.read2genome(df_train)
    log.write("Prediction on validation")
    df_valid_pd = r2g.read2genome(df_valid)
    # If tax_level different than species, we have to change the y true
    range_threshold = np.arange(0, 1, 0.01)[::-1]

    if (
        read2genome == "transformer" or read2genome == "fastDNA"
    ) and tax_level != tax_id_name:
        D_convert = dict(zip(df_taxonomy_ref[ncbi_id_name], df_taxonomy_ref[tax_level]))
        df_train_pd[tax_level] = df_train_pd[tax_id_name].apply(
            lambda x: int(D_convert[str(x)])
        )
        df_valid_pd[tax_level] = df_valid_pd[tax_id_name].apply(
            lambda x: int(D_convert[str(x)])
        )

    # Computing scores
    df_count_label_train = pd.DataFrame(df_train_pd[tax_level].value_counts())
    log.write("Computing metric scores on train")
    save_score(
        path_save,
        df_train_pd,
        df_count_label_train,
        threshold=0.0,
        dataset="train",
        file_mode="w",
    )
    log.write("Computing metric scores by threshold on train")
    (
        A_reject_rate,
        A_accuracy,
        A_precision,
        A_recall,
        A_f1_score,
        A_count_reject,
    ) = compute_metrics_by_threshold(df_train_pd, df_count_label_train, range_threshold)
    log.write("Plot scores train")
    plot_score(
        os.path.join(path_save, "metrics_by_threshold_train.png"),
        range_threshold,
        A_reject_rate,
        A_accuracy,
        A_precision,
        A_recall,
        A_f1_score,
        "Metrics on train dataset by reject threshold",
    )
    plot_score2(
        os.path.join(path_save, "metrics_by_reject_rate_train.png"),
        A_reject_rate,
        A_accuracy,
        A_precision,
        A_recall,
        A_f1_score,
        "Metrics on train dataset by reject rate",
    )
    log.write("Plot reject train")
    plot_reject_by_abundance(
        os.path.join(path_save, "reject_rate_train.png"),
        df_count_label_train,
        A_count_reject,
        D_id,
        "Rejected rate on train dataset",
    )
    log.write("Plot heatmap train")
    heatmap(
        df_train_pd[tax_level],
        df_train_pd[pred_name],
        D_id,
        path_save,
        "train",
        "Heatmap classification rate on train dataset",
    )
    log.write("Computing Bowtie score and scores with best threshold on train")
    best_threashold = range_threshold[
        np.argmax(A_precision / (A_reject_rate + 1) ** 3)
    ]  # precision / (reject_rate + 1)^3
    if bowtie_index is not None or bowtie_index == "None":
        save_bowtie_score(
            os.path.join(path_save, "bowtie_score.csv"),
            df_train_pd,
            best_threashold,
            path_tmp_folder,
            n_instance,
            "train",
            file_mode="w",
        )
    log.write("Computing with best threshold on train")
    save_score(
        path_save,
        df_train_pd,
        df_count_label_train,
        threshold=best_threashold,
        dataset="train",
        file_mode="a",
    )
    del df_train_pd

    df_count_label_valid = pd.DataFrame(df_valid_pd[tax_level].value_counts())
    log.write("Computing metric scores on valid")
    save_score(
        path_save, df_valid_pd, df_count_label_valid, dataset="valid", file_mode="a"
    )
    log.write("Computing metric scores by threshold on valid")
    df_count_label = pd.DataFrame(df_valid_pd[tax_level].value_counts())
    (
        A_reject_rate,
        A_accuracy,
        A_precision,
        A_recall,
        A_f1_score,
        A_count_reject,
    ) = compute_metrics_by_threshold(df_valid_pd, df_count_label, range_threshold)
    log.write("Plot scores valid")
    plot_score(
        os.path.join(path_save, "metrics_by_threshold_valid.png"),
        range_threshold,
        A_reject_rate,
        A_accuracy,
        A_precision,
        A_recall,
        A_f1_score,
        "Metrics on valid dataset by reject threshold",
    )
    plot_score2(
        os.path.join(path_save, "metrics_by_reject_rate_valid.png"),
        A_reject_rate,
        A_accuracy,
        A_precision,
        A_recall,
        A_f1_score,
        "Metrics on valid dataset by reject rate",
    )
    log.write("Plot reject valid")
    plot_reject_by_abundance(
        os.path.join(path_save, "reject_rate_valid.png"),
        df_count_label,
        A_count_reject,
        D_id,
        "Rejected rate on valid dataset",
    )
    log.write("Plot heatmap valid")
    heatmap(
        df_valid_pd[tax_level],
        df_valid_pd[pred_name],
        D_id,
        path_save,
        "valid",
        "Heatmap classification rate on valid dataset",
    )
    log.write("Computing Bowtie score with best threshold on valid")

    best_threashold = range_threshold[
        np.argmax(A_precision / (A_reject_rate + 1) ** 3)
    ]  # precision / (reject_rate + 1)^3
    if os.getenv("BOWTIE") is not None:
        save_bowtie_score(
            os.path.join(path_save, "bowtie_score.csv"),
            df_valid_pd,
            best_threashold,
            path_tmp_folder,
            n_instance,
            "valid",
            file_mode="a",
        )
    log.write("Computing with best threshold on valid")
    save_score(
        path_save,
        df_valid_pd,
        df_count_label_valid,
        threshold=best_threashold,
        dataset="valid",
        file_mode="a",
    )

    log.write("Computing plot true proportion vs pred proportion")
    df_valid = spark.createDataFrame(df_valid_pd)
    del df_valid_pd
    win_sim = Window.partitionBy(df_valid[sim_id_name])
    df_proportion_pred = df_valid.groupBy(sim_id_name, pred_name).count()
    df_proportion_pred = (
        df_proportion_pred.withColumn(
            prop_pred_name, F.col("count") / F.sum("count").over(win_sim)
        )
        .select(sim_id_name, pred_name, prop_pred_name)
        .withColumnRenamed(pred_name, tax_level)
    )
    df_proportion = df_valid.groupBy(sim_id_name, tax_level).count()
    df_proportion = df_proportion.withColumn(
        prop_true_name, F.col("count") / F.sum("count").over(win_sim)
    ).select(sim_id_name, tax_level, prop_true_name)
    df_proportion = df_proportion.join(
        df_proportion_pred, on=[sim_id_name, tax_level], how="fullouter"
    )
    df_proportion = df_proportion.toPandas()
    df_proportion = df_proportion.fillna(0)
    sim_ids = df_proportion[sim_id_name].values

    path_proportion = os.path.join(path_save, "plot_proportion_valid")
    hdfs.create_dir(path_proportion, mode="local")
    for sim_id in sim_ids:
        path_save_ = os.path.join(path_proportion, sim_id + ".png")
        plot_proportion_true_pred(
            path_save_,
            df_proportion,
            sim_id,
            "Plot between true and predicted abundance proportion for simulation %s"
            % sim_id,
        )

    log.write("Saving the correlation matrix")
    create_correlation_table(
        os.path.join(path_save, "proportion_correlation_table_valid.csv"), df_proportion
    )
    log.write("Analyse finished")
    log.close()
