import os
import sys
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib

matplotlib.use("agg")

root_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, os.path.join(root_folder, "utils"))
sys.path.insert(0, os.path.join(root_folder, "read2vec"))
sys.path.insert(0, os.path.join(root_folder, "read2genome"))

import parser_creator
import hdfs_functions as hdfs
from string_names import *


def kmer_count(path_data, path_save, mode="word_count_matrix"):
    if mode == "word_count_matrix":
        spark = hdfs.createSparkSession("Analyse word count")
        df = spark.read.parquet(path_data)
        df_pd = df[[kmer_name, count_name]].toPandas()
        occurrence = np.array(df_pd[count_name])
        labels = np.array(df_pd[kmer_name])
        del df_pd

    elif mode == "kmerized_genome_dataset":
        with open(path_data, "r") as f:
            counter = Counter()
            for line in tqdm(f):
                counter.update(line.replace("\n", "").split())
        most_common = counter.most_common()

        most_common = list(filter(lambda x: x[1] not in ["<unk>", "</s>"], most_common))
        occurrence = np.array([x[1] for x in most_common])
        labels = np.array([x[0] for x in most_common])
    else:
        raise Exception("mode doesn't exist")
    os.makedirs(path_save, exist_ok=True)
    if len(labels) < 5000:
        plot(os.path.join(path_save, "plot.png"), occurrence, labels)
    pd.DataFrame(
        np.array([labels, occurrence]).T, columns=[kmer_name, count_name]
    ).to_csv(os.path.join(path_save, "matrix.csv"), index=False)


def plot(path_save, occurrence, labels):
    fig, ax = plt.subplots(figsize=(10, 10))  # in inches
    ax.set_xticklabels(labels, rotation="vertical")
    ax.bar(range(len(occurrence)), occurrence)
    ax.set_xlabel("kmers")
    ax.set_ylabel("occurrence")
    ax.set_title("kmer count")
    plt.savefig(path_save)
    plt.tight_layout()
    plt.close()


if __name__ == "__main__":
    parser = parser_creator.ParserCreator()
    args = parser.parser_kmer_count()
    path_data = args.path_data
    path_save = args.path_save
    mode = args.mode
    kmer_count(path_data, path_save, mode)
