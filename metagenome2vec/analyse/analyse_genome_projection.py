import pandas as pd
import numpy as np
import sys
import os
import re
from sklearn.manifold import TSNE
from umap import UMAP
from pyspark.sql import types as T
from pyspark.sql import functions as F
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn
seaborn.set_style("whitegrid")
import matplotlib.pylab as pylab
import matplotlib.cm as cm
from scipy.spatial.distance import pdist, squareform
from skbio.stats.distance import mantel
import time
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

root_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, os.path.join(root_folder, "utils"))
import parser_creator
import logger
import data_manager
import hdfs_functions as hdfs
from string_names import *


params = {'legend.fontsize': 'x-large',
          'figure.figsize': (10, 8),
          'axes.labelsize': 'x-large',
          'axes.titlesize': 'x-large',
          'figure.titlesize': 'x-large'}
pylab.rcParams.update(params)


# Visualize the embeddings and save results.
def plot_scatters(low_dim_embs, y_, tax_level, title=None, path_save=None, metadata=None):
    """
    Plot the t-SNE projection with heat map according to the occurence of each k-mer
    :param low_dim_embs: numpy 2D array, embeddings of the t-sne project
    :param y_: pandas dataframe, contains tax level columns of low_dim_embs
    :param tax_level: str, the tax level
    :param title: str, title of the plot
    :param path_save: str, path where is save the figure
    :param: metadata: Pandas Dataframe, dataframe containing metadata about genomes
    :return:
    """
    fig, ax = plt.subplots(figsize=(18, 18))  # in inches
    x, y = low_dim_embs[:, 0], low_dim_embs[:, 1]
    D_tax_point = {}
    D_tax_color = {}
    L_label = []  # Just to keep the right order
    colors = cm.rainbow(np.linspace(0, 1, len(set(y_[tax_level].tolist()))))

    cpt = 0
    for i, label in enumerate(y_[tax_level]):
        if label not in D_tax_point:
            L_label.append(label)
            D_tax_point[label] = []
            D_tax_color[label] = colors[cpt]
            cpt += 1
        D_tax_point[label].append(i)

    for tax in L_label:
        if metadata is not None:
            try:
                label = metadata[metadata[tax_level] == str(tax)].iloc[0][tax_level + "_name"]
                label = "Not defined" if label == "-1" else label
            except:
                label = "Not defined"
        else:
            label = tax
        ax.scatter(x[D_tax_point[tax]], y[D_tax_point[tax]],
                   label=label, c=D_tax_color[tax])

    ax.legend(bbox_to_anchor=(1.2, 1.), ncol=int(len(colors) / 20))
    if title is not None:
        ax.set_title(title, size=25)
    if path_save is not None:
        fig.savefig(path_save)


def clean_and_split(x, max_length):
    x = re.sub("NN+", "", x).replace("\n", "")
    return re.findall(".{%s}?" % max_length, x)

# * k because the sequence is cut into kmer max_length times
udfCleanAndSplit = F.udf(lambda x: clean_and_split(x, max_length), T.ArrayType(T.StringType()))


def compute(path_data, path_metadata, path_save):
    col_name = "sequence"
    schema = T.StructType([T.StructField(col_name, T.StringType(), False)])
    #L_perplexity = [5, 20, 50, 100]
    L_n_neighbors = [5, 20, 50, 100]
    metric = "correlation"
    min_dist = 0.3
    #n_iter = 2000
    #learning_rate = 400
    metadata = pd.read_csv(path_metadata).astype(str)
    if overwrite is False and os.path.exists(os.path.join(path_save, "genome_embeddings.csv")):
        X = pd.read_csv(os.path.join(path_save, "genome_embeddings.csv"))
    else:
        X = None
    for i, fasta in enumerate(metadata[fasta_name].tolist()):
        metadata_row = metadata.iloc[i]
        # test if fasta already computed in X
        if not overwrite and X is not None and metadata_row[fasta_name] in X[fasta_name].tolist():
            continue
        log.writeExecutionTime()
        # read
        df_genome = spark.read.csv(os.path.join(path_data, fasta), schema=schema)
        # filter only sequence
        d = time.time()
        df_genome = df_genome.filter(~df_genome.sequence.rlike("^>"))
        # cleaning, splitting and exploding
        df_genome = df_genome.withColumn(col_name, udfCleanAndSplit(col_name))
        df_genome = df_genome.withColumn(col_name, F.explode(df_genome.sequence))
        # reapartition to use more cpu
        df_genome = df_genome.repartition(num_partitions).persist()
        log.write("Nb row: %s " % df_genome.count())
        log.write("cleaning done in: %s" % (time.time() - d))
        # Run read2vec
        d = time.time()
        df_genome = r2v.read2vec(df_genome, col_name=col_name)
        log.write("read2vec done in: %s" % (time.time() - d))
        # Create new row for X (res)
        d = time.time()
        new_row = pd.Series(df_genome.groupBy().mean().toPandas().values[0])
        new_row = pd.DataFrame(pd.concat([metadata_row, new_row])).T
        new_row.columns = new_row.columns.astype(str)
        # concatenate with previous X
        X = pd.concat([X, new_row]) if X is not None else new_row
        log.write("add res in X done in: %s" % (time.time() - d))
        # saving in csv file
        X.to_csv(os.path.join(path_save, "genome_embeddings.csv"), index=False)
        spark.catalog.clearCache()
        for (id, rdd) in spark.sparkContext._jsc.getPersistentRDDs().items():
            rdd.unpersist()
        log.writeExecutionTime("%s : %s" % (i, fasta))
    # Remove .fna because JolyTree fasta name are without .fna
    metadata[fasta_name] = metadata[fasta_name].apply(lambda x: x.replace(".fna", ""))
    y_ = X[[x for x in X.columns.values if not x.isnumeric()]]
    X = X.drop([x for x in X.columns.values if not x.isnumeric()], axis=1)
    for tax_level in [species_name, genus_name, family_name]:
        log.write("t-SNE on %s level" % tax_level)
        for n_neighbors in L_n_neighbors:
            path_save_fig = os.path.join(path_save, "UMAP_tax_level_%s_n_neighbors_%s_min_dist_%s_metric_%s" % (tax_level, n_neighbors, str(min_dist).replace('.', '_'), metric))
            #path_save_fig = os.path.join(path_save, "t_SNE_tax_level_%s_perp_%s_iter_%s_l_rate_%s" % (tax_level, perplexity, n_iter, learning_rate))
            log.write("Computing %s" % path_save_fig)
            if overwrite is False and os.path.exists(path_save_fig):
                continue
            umap = UMAP(n_neighbors=n_neighbors, n_components=2,
                        min_dist=min_dist, metric=metric)
            low_dim_embs = umap.fit_transform(X)
            title = "UMAP projection with tax level = %s, # neighbors = %s, min dist = %s, metric = %s\n" % (tax_level, n_neighbors, min_dist, metric)
            # tsne = TSNE(perplexity=perplexity, n_components=2,
            #             n_iter=n_iter, learning_rate=learning_rate)
            # low_dim_embs = tsne.fit_transform(X)
            #title = "t-SNE projection with tax level = %s, perplexity = %s, iterations = %s, learning rate = %s\n" % (tax_level, perplexity, n_iter, learning_rate)
            plot_scatters(low_dim_embs, y_, tax_level, title=title, path_save=path_save_fig, metadata=metadata)
    # Compute similarity
    log.write(path_genome_dist)
    if path_genome_dist is not None:
        try:
            genome_dist = pd.read_csv(path_genome_dist, header=None, sep=",", skiprows=1)
            # Reorder the matrix to correspond to metadata file
            genome_dist = genome_dist.reset_index()
            genome_dist[index_name] = genome_dist[index_name] + 1
            # col 0 correspond to fasta name from metadata
            genome_dist = genome_dist.set_index(0).reindex(index=metadata[fasta_name])
            genome_dist = genome_dist[genome_dist[index_name].values]
            X_dist = squareform(pdist(X.iloc[:, 4:], metric="cosine"))
            score, p_value, _ = mantel(genome_dist, X_dist, method="spearman")
            log.write("Mantel score={:.2f} ; p_value={:.5f}".format(score, p_value))
            with open(os.path.join(path_save, "mantel_test.txt"), "w") as f_out:
                f_out.write("Mantel score={:.2f} ; p_value={:.5f}".format(score, p_value))
        except:
            pass


if __name__ == '__main__':
    parser = parser_creator.ParserCreator()
    args = parser.parser_genome_projection()
    overwrite = args.overwrite
    path_read2vec = args.path_read2vec
    read2vec = args.read2vec
    path_data = args.path_data
    n_cpus = args.n_cpus
    path_metagenome_word_count = args.path_metagenome_word_count
    k = args.k_mer_size
    id_gpu = [int(x) for x in args.id_gpu.split(",")]
    path_tmp_folder = args.path_tmp_folder
    path_save = args.path_save
    hdfs.create_dir(path_save, mode="local")
    path_metadata = args.path_metadata
    path_genome_dist = None if args.path_genome_dist=="None" else args.path_genome_dist
    max_length = args.max_length
    log_file = args.log_file
    path_log = args.path_log
    num_partitions = args.num_partitions

    log = logger.Logger(path_log, log_file, log_file,
                        variable_time={"read2vec": read2vec},
                        **vars(args))

    # Init Spark
    spark = hdfs.createSparkSession("genome_projection")

    # Preparing read2vec
    log.write("Loading read2vec")
    r2v = data_manager.load_read2vec(read2vec, path_read2vec=path_read2vec, spark=spark,
                                     path_metagenome_word_count=path_metagenome_word_count, k=k, id_gpu=id_gpu,
                                     path_tmp_folder=path_tmp_folder)
    
    # Compute t-SNE projection
    log.write("Computing")
    compute(path_data, path_metadata, path_save)

