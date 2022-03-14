import numpy as np
import sys
import os
from multiprocessing import Pool
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import matplotlib
from collections import Counter

matplotlib.use('agg')
root_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, os.path.join(root_folder, "utils"))
import parser_creator
import data_manager
if sys.version_info[0] == 3 and sys.version_info[1] == 7:
    import transformation_ADN
else:
    import transformation_ADN2 as transformation_ADN
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn
seaborn.set_style("whitegrid")
import matplotlib.pylab as pylab

params = {'legend.fontsize': 'x-large',
          'figure.figsize': (10, 8),
          'axes.labelsize': 'x-large',
          'axes.titlesize': 'x-large',
          'figure.titlesize': 'x-large'}
pylab.rcParams.update(params)

###################################
#---------------------------------#
#------- Usefull functions -------#
#---------------------------------#
###################################


# Visualize the embeddings and save results.
def plot_with_labels(low_dim_embs, reverse_index, kmer_count, heat_map=False, cmap=None, quantiles=None, show_label=False, title=None, path_save=None):
    """
    Plot the t-SNE projection with heat map according to the occurence of each k-mer
    :param low_dim_embs: numpy 2D array, embeddings of the t-sne project
    :param reverse_index: dict, reverse index for kmer : {index: kmer}
    :param heat_map: boolean, if true compute an heat map
    :param cmap:
    :param quantiles: numpy array, used to compute color
    :param show_label: boolean, if True plot the value of the kmer
    :param title: str, title of the plot
    :param path_save: str, path where is save the figure
    :return:
    """
    def fmt(x):
        a, b = '{:.2e}'.format(x).split('e')
        b = int(b)
        return r'${} \times 10^{{{}}}$'.format(a, b)
    fig, ax = plt.subplots(figsize=(18, 18))  # in inches
    if heat_map:
        colors = cmap(np.linspace(0, 1, len(quantiles) + 1))
    x, y = low_dim_embs[:, 0], low_dim_embs[:, 1]
    if heat_map:
        colors_data = []
        for i, label in reverse_index.items():
            try:
                colors_data.append(colors[len(np.where(kmer_count[label] >= quantiles)[0])])
            except:
                pass
    else:
        colors_data = None
    ax.scatter(x, y, color=colors_data)

    if show_label:
        for i, label in reverse_index.items():
            try:
                x, y = low_dim_embs[i, :]
                ax.annotate(label,
                            xy=(x, y),
                            xytext=(5, 2),
                            textcoords='offset points',
                            ha='right',
                            va='bottom')
            except:
                pass

    ax.tick_params(axis='both', which='major', labelsize=18)
    if title is not None:
        ax.set_title(title, size=25)
    if heat_map:
        cax, _ = matplotlib.colorbar.make_axes(ax)
        normalize = matplotlib.colors.Normalize(vmin=min(quantiles), vmax=max(quantiles))
        cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm=normalize)  #, format=ticker.FuncFormatter(fmt))
        cbar.ax.set_title("K-mers' occurrence")
    if path_save is not None:
        fig.savefig(path_save)

#########################################################################################################


def generator(similarities, reverse_index):
    for i in range(len(similarities)):
        for j in range(i + 1, len(similarities)):
            kmer_1, kmer_2 = reverse_index[i], reverse_index[j]
            if kmer_1 == kmer_2:
                continue
            else:
                yield (i, j)


def dist_ED(i, j):
    return i, j, transformation_ADN.ED(reverse_index[i], reverse_index[j])


def dist_NW(i, j):
    return i, j, transformation_ADN.needle(reverse_index[i], reverse_index[j])


def create_distance(similarities, dist, reverse_index):
    """
    Compute the distance relation between edit or needleman wusch distance and cosine similarity
    :param similarities: numpy 2D array, matrix of the cosine similarity
    :param algo: func, edit or needleman-wunch
    :param reverse_index: dict, reverse index for kmer : {index: kmer}
    :return: D_sistance, dict Key is the distance value, value is the list of all cosine similarity
    """
    gen = generator(similarities, reverse_index)
    D_distance = {}
    pool = Pool(n_cpus)
    res = pool.starmap(dist, gen)
    pool.close()
    pool.join()
    for tup in res:
        sim = similarities[tup[0], tup[1]]
        try:
            D_distance[tup[2]].append(sim)
        except:
            D_distance[tup[2]] = [sim]
    return D_distance

#########################################################################################################


def compute(path_embeddings, embeddings, index, reverse_index):
    if n_instance is not None:
        index_to_keep = []
        kmer_count2 = {}

        for i in range(0, n_instance):
            try:
                index_to_keep.append(index[kmer_index[i]])
                kmer_count2[kmer_index[i]] = kmer_count[kmer_index[i]]
            except:
                continue
        embeddings = embeddings[index_to_keep]
        del index_to_keep
        del index
    else:
        del index
        kmer_count2 = kmer_count.copy()
    # t-SNE
    if any([x in computation_type for x in [0, 3]]):
        quantiles = np.quantile(np.array(list(kmer_count2.values()), dtype=int), np.arange(0.1, 1., 0.1))
        cmap = matplotlib.cm.get_cmap('rainbow')
        L_perplexity = [5, 20, 50, 100]
        n_iter = 2000
        learning_rate = 200
        for perplexity in L_perplexity:
            if not overwrite and os.path.exists(os.path.join(path_embeddings, "t_SNE_perp_%s_iter_%s_l_rate_%s" % (perplexity, n_iter, learning_rate))):
                continue
            tsne = TSNE(perplexity=perplexity, n_components=2,
                        n_iter=n_iter, learning_rate=learning_rate)
            low_dim_embs = tsne.fit_transform(embeddings)
            title = "t-SNE projection with perplexity = %s, iterations = %s, learning rate = %s\n" % (perplexity, n_iter, learning_rate)
            plot_with_labels(low_dim_embs, reverse_index, kmer_count2, cmap=cmap, quantiles=quantiles,
                             heat_map=True, show_label=False, title=title,
                             path_save=os.path.join(path_embeddings, "t_SNE_perp_%s_iter_%s_l_rate_%s" % (perplexity, n_iter, learning_rate)))
    # Edit distance vs cosine similarity
    try:
        if any([x in computation_type for x in [1, 2, 3]]):
            sparse_embeddings = sparse.csr_matrix(embeddings)
            similarities = cosine_similarity(sparse_embeddings)
    except:
        pass
    try:
        if any([x in computation_type for x in [1, 3]]):
            if overwrite or os.path.exists(os.path.join(path_embeddings, "cosine_vs_edit")) is False:
                D_ld = create_distance(similarities, dist_ED, reverse_index)
                fig, axe = plt.subplots(figsize=(8, 8))
                scores = list(D_ld.keys())
                pos = [i for i in range(min(scores), max(scores) + 1)]
                axe.violinplot([D_ld[i] for i in range(min(scores), max(scores) + 1)], pos, points=60, widths=0.7, showmeans=True,
                               showextrema=True, showmedians=True, bw_method=0.5)
                axe.set_title('Cosine similarity vs Edit Distance', fontsize=25)
                axe.set_xlabel("Edit Distance", fontsize=18)
                axe.set_ylabel("Cosine Similarity", fontsize=18)
                axe.set_xticks(pos)
                axis = np.concatenate([np.repeat(k, len(D_ld[k])) for k in D_ld.keys()])
                ordinate = np.concatenate([D_ld[k] for k in D_ld.keys()])
                seaborn.regplot(x=axis, y=ordinate, order=2, x_jitter=.05, ax=axe, scatter=False)
                axe.figure.savefig(os.path.join(path_embeddings, "cosine_vs_edit"))
    except:
        pass
    try:
        if any([x in computation_type for x in [2, 3]]):
            if overwrite or os.path.exists(os.path.join(path_embeddings, "cosine_vs_needleman-wunsch")) is False:
                D_needle = create_distance(similarities, dist_NW, reverse_index)
                fig, axe = plt.subplots(figsize=(8, 8))
                scores = list(D_needle.keys())
                pos = [i for i in range(min(scores), max(scores) + 1)]
                axe.violinplot([D_needle[i] for i in range(min(scores), max(scores) + 1)], pos, points=60, widths=0.7, showmeans=True,
                               showextrema=True, showmedians=True, bw_method=0.5)
                axe.set_title('Cosine similarity vs Needleman-Wunsch Score', fontsize=25)
                axe.set_xlabel("Needleman-Wunsch Score", fontsize=18)
                axe.set_ylabel("Cosine Similarity", fontsize=18)
                axe.set_xticks(pos)
                axis = np.concatenate([np.repeat(k, len(D_needle[k])) for k in D_needle.keys()])
                ordinate = np.concatenate([D_needle[k] for k in D_needle.keys()])
                seaborn.regplot(x=axis, y=ordinate, order=2, x_jitter=.05, ax=axe, scatter=False)
                axe.figure.savefig(os.path.join(path_embeddings, "cosine_vs_needleman-wunsch"))
    except:
        pass


if __name__ == '__main__':
    parser = parser_creator.ParserCreator()
    args = parser.parser_analyse_embeddings()
    overwrite = args.overwrite
    path_kmer2vec = args.path_kmer2vec
    path_data = args.path_data

    computation_type = [int(x) for x in args.computation_type.split(',') if int(x) in [0, 1, 2, 3]]
    n_cpus = args.n_cpus
    n_instance = args.n_instance  # Number of k-mers to compute

    if any([x in computation_type for x in [0, 3]]):
        with open(path_data, "r") as f:
            counter = Counter()
            for line in f:
                counter.update(line.replace("\n", "").split())
        most_common = counter.most_common()
        most_common = list(filter(lambda x: x[1] not in ["<unk>", "</s>"], most_common))
        kmer_count = {x[0]: x[1] for x in most_common}
        kmer_index = {i: x[0] for i, x in enumerate(most_common)}
    embeddings, index, reverse_index = data_manager.load_embeddings(path_kmer2vec,
                                                                    skip_kmer_name=True,
                                                                    L_kmer_to_del=["<unk>", "</s>"])

    compute(path_kmer2vec, embeddings, index, reverse_index)


