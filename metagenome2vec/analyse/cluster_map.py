import pandas as pd
import numpy as np
import re
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist, squareform
import sys
import os
import matplotlib
import seaborn

seaborn.set()
import matplotlib.pylab as pylab

params = {
    "legend.fontsize": "x-large",
    "figure.figsize": (15, 12),
    "axes.labelsize": "x-large",
    "axes.titlesize": "x-large",
    "figure.titlesize": "x-large",
}
pylab.rcParams.update(params)
matplotlib.use("agg")
root_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, os.path.join(root_folder, "utils"))
import parser_creator
from string_names import *


###################################
# ---------------------------------#
# ------- Usefull functions -------#
# ---------------------------------#
###################################


def cluster_map(path_data, path_metadata):
    """
    Compute the cluster map
    :param path_data: String, complete path to the matrix
    :param path_data: String, complete path to the metadata
    :return:
    """
    name_file_res = "_".join(
        [re.sub("^(.*)\..*$", "\\1", path_data), "cluster_map.png"]
    )
    if os.path.isfile(path_metadata):
        meta_info = pd.read_csv(path_metadata, delimiter=",")[
            [id_fasta_name, group_name]
        ]
    else:
        meta_info = None
    X = pd.read_csv(path_data)
    n_clusters = pd.unique(
        X[id_fasta_name].apply(lambda x: re.sub("_[0-9]+", "", x))
    ).shape[0]
    agglo = AgglomerativeClustering(
        n_clusters=n_clusters, affinity="cosine", linkage="complete"
    )
    clusters = agglo.fit_predict(X.iloc[:, 2:])  # don't take name and type
    print(clusters)
    cluster_map = seaborn.clustermap(
        pd.DataFrame(
            squareform(pdist(X.iloc[:, 2:], metric="cosine")),
            index=X[id_fasta_name],
            columns=X[id_fasta_name],
        ),
        cmap="Blues_r",
        xticklabels=1,
        yticklabels=1,
    )
    if meta_info is not None:
        for y_tick_label, x_tick_label in zip(
            cluster_map.ax_heatmap.axes.get_yticklabels(),
            cluster_map.ax_heatmap.axes.get_xticklabels(),
        ):
            y_tick_text = y_tick_label.get_text()
            x_tick_text = x_tick_label.get_text()
            try:
                if (
                    meta_info[
                        meta_info[id_fasta_name] == re.sub("__[0-9]*$", "", y_tick_text)
                    ][group_name]
                    == control_category
                ).values[0]:
                    y_tick_label.set_color("blue")
                else:
                    y_tick_label.set_color("red")
                if (
                    meta_info[
                        meta_info[id_fasta_name] == re.sub("__[0-9]*$", "", x_tick_text)
                    ][group_name]
                    == control_category
                ).values[0]:
                    x_tick_label.set_color("blue")
                else:
                    x_tick_label.set_color("red")
            except:
                import pdb

                pdb.set_trace()
        cluster_map.ax_heatmap.axes.set_xticks(
            np.arange(0, len(cluster_map.ax_heatmap.axes.get_xticklabels()), 1.0)
        )
        cluster_map.ax_heatmap.axes.set_yticks(
            np.arange(0, len(cluster_map.ax_heatmap.axes.get_yticklabels()), 1.0)
        )
    print(name_file_res)
    cluster_map.savefig(name_file_res)


if __name__ == "__main__":
    parser = parser_creator.ParserCreator()
    args = parser.parser_cluster_map()
    path_metadata = args.path_metadata
    path_data = args.path_data

    cluster_map(path_data, path_metadata)
