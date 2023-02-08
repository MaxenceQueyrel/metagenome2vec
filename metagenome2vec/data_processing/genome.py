import pandas as pd
from Bio import SeqIO
import requests
from bs4 import BeautifulSoup
import os
import time
import re
from tqdm import tqdm
from ete3 import NCBITaxa

from metagenome2vec.utils import file_manager, transformation_ADN
from metagenome2vec.utils.string_names import *


# Functions
def get_fasta_file(path_data):
    """
    Return the list of all fasta files in a folder
    :param path_data: String, complete path to the folder containing fasta file
    :return: List of str, all the file_name in the folder
    """
    # Get all the
    L_fasta_file = []
    for (_, _, filenames) in os.walk(path_data):
        L_fasta_file.extend(
            list(
                filter(lambda x: x.endswith(".fna") or x.endswith(".fasta"), filenames)
            )
        )
        break
    return L_fasta_file


def count_bp_length(path_fasta_file):
    """
    Count the base pair lenght of a fasta file
    :param path_fasta_file: String, complete path to the fasta file
    :return: int, bp_length of the fasta file
    """
    with open(path_fasta_file, "r") as fasta_file:
        total_length = 0
        for cur_record in SeqIO.parse(fasta_file, "fasta"):
            total_length += (
                cur_record.seq.count("A")
                + cur_record.seq.count("C")
                + cur_record.seq.count("G")
                + cur_record.seq.count("T")
            )
    return total_length


def get_taxonomy_id(path_fasta_file):
    """
    Return the taxonomic id of a fasta file by scraping the ncbi website
    :param path_fasta_file: String, complete path to the fasta file
    :return: int, tax id
    """
    tax_id = None
    with open(path_fasta_file, "r") as fasta_file:
        # get the id of the fasta
        for cur_record in SeqIO.parse(fasta_file, "fasta"):
            id_ = cur_record.id.rsplit(".", 1)[0]
            break
        nb_try = 2  # number of tries to get html page
        cpt = 0  # tries count
        while cpt < nb_try * 6:
            tax_link = ""
            # Change the id if it doesn't work after nb_try
            if cpt == nb_try or cpt == nb_try * 4:
                id_ += ".1"
            elif cpt == nb_try * 2 or cpt == nb_try * 5:
                id_ = id_.replace(".1", ".2")
            elif cpt == nb_try * 3:
                id_ = id_.rsplit(".", 1)[0].replace("GCA", "GCF")
            # Get the taxonomic link
            if tax_link == "":
                try:
                    page = requests.get(
                        "https://www.ncbi.nlm.nih.gov/assembly/{}/".format(id_)
                    )
                    soup = BeautifulSoup(page.content, "html.parser")
                    tax_link = "https://www.ncbi.nlm.nih.gov/" + re.sub(
                        '.*"(.*)".*',
                        "\\1",
                        str(
                            soup.find(id="summary").findAll(
                                "a", href=re.compile("^/Taxonomy.*")
                            )[0]
                        ),
                    ).replace("&amp;", "&")
                    cpt = 0
                except:
                    cpt += 1
                    time.sleep(0.5)
                    continue
            # Get the taxonomic id
            try:
                page = requests.get(tax_link)
                soup = BeautifulSoup(page.content, "html.parser")
                tax_id = re.search("Taxonomy ID: ([0-9]*)", str(soup))[1]
            except:
                cpt += 1
                time.sleep(0.5)
                continue
            break
    assert tax_id is not None, "{} failed to output correct taxonomic id".format(id_)
    return tax_id


def get_ranks(taxid, L_rank, ncbi):
    """
    Return a dictionary with all ranks of L_rank, from ncbi database, for the taxon with id equals to taxid
    :param taxid: int, the taxon id
    :param L_rank: List of str containing phylogenetic rank
    :param ncbi: NCBI object used to call ncbi API
    :return: Dictionary {rank_id: rank} ex {species_id: 123456}
    """
    lineage = ncbi.get_lineage(taxid)
    lineage2ranks = ncbi.get_rank(lineage)
    ranks2lineage = dict((rank, taxid) for (taxid, rank) in lineage2ranks.items())
    return {
        "{}_id".format(rank): ranks2lineage.get(rank, "<not present>")
        for rank in L_rank
    }


def complete_df_with_all_phylo_id(df, taxa_id="NCBI_ID"):
    """
    Complete a dataframe with several phylogenetic ranks from a column with taxa ids
    :param df: Pandas Dataframe, with one column containing taxa ids
    :param taxa_id: String, name of the column containing taxa ids
    :return:
    """
    L_rank = [
        kingdom_name,
        phylum_name,
        class_name,
        order_name,
        family_name,
        genus_name,
        species_name,
        strain_name,
    ]
    ncbi = NCBITaxa()
    for i, tax in tqdm(df[taxa_id].iteritems()):
        D_res = get_ranks(int(tax), L_rank, ncbi)
        for rank in L_rank:
            df.loc[i, rank] = (
                int(D_res[rank + "_id"])
                if D_res[rank + "_id"] != "<not present>"
                else -1
            )
        for rank in L_rank:
            df.loc[i, rank + "_name"] = (
                ncbi.translate_to_names([int(D_res[rank + "_id"])])[0]
                if D_res[rank + "_id"] != "<not present>"
                else -1
            )
    for rank in L_rank:
        df[rank] = df[rank].apply(int)
    for i in range(2, len(L_rank) + 1):
        j = len(L_rank) - i
        index = df[df[L_rank[j]] == -1].index
        if len(index) > 0:
            df.loc[index, L_rank[j]] = df.loc[index, L_rank[j + 1]]
            df.loc[index, L_rank[j] + "_name"] = df.loc[index, L_rank[j + 1] + "_name"]


def create_df_fasta_metadata(path_fasta_folder):
    """
    Construct the metadata dataframe of a collection of fasta file located in path_data
    :param path_fasta_folder: String, complete path to the folder containing fasta file
    :return: DataFrame Pandas, the complete metadata fatadrame
    """
    L_fasta_file = get_fasta_file(path_fasta_folder)
    df_fasta_metadata = pd.DataFrame(L_fasta_file, columns=[fasta_name])
    df_fasta_metadata[genome_id_name] = [
        "genome_{}".format(x) for x in range(len(df_fasta_metadata))
    ]
    df_fasta_metadata[novelty_category_name] = known_strain_name
    df_fasta_metadata[OTU_name] = [x for x in range(len(df_fasta_metadata))]
    df_fasta_metadata[ncbi_id_name] = [
        get_taxonomy_id(os.path.join(path_fasta_folder, fasta_file))
        for fasta_file in tqdm(L_fasta_file)
    ]
    # Change ncbi id if same
    """
    D_ncbi_id = {}
    for i in range(df_fasta_metadata[ncbi_id_name].shape[0]):
        ncbi_id = df_fasta_metadata[ncbi_id_name].iloc[i]
        if ncbi_id not in D_ncbi_id:
            D_ncbi_id[ncbi_id] = 0
        else:
            df_fasta_metadata[ncbi_id_name].iloc[i] = "%s_%s" % (ncbi_id, str(D_ncbi_id[ncbi_id]))
            D_ncbi_id[ncbi_id] += 1
    """
    df_fasta_metadata[bp_length_name] = [
        count_bp_length(os.path.join(path_fasta_folder, fasta_file))
        for fasta_file in tqdm(L_fasta_file)
    ]
    complete_df_with_all_phylo_id(df_fasta_metadata)
    df_fasta_metadata = df_fasta_metadata.rename(columns={"strain_name": tax_name})
    return df_fasta_metadata


def kmerization(path_data, path_save, k_mer_size, step):
    file_manager.create_dir(os.path.dirname(path_save), "local")
    k = k_mer_size
    s = step

    n_lines = sum(1 for _ in open(path_data, "r"))
    with open(path_data, "r") as genome:
        with open(path_save, "w") as f_res:
            for line in tqdm(genome, total=n_lines):
                if line[0] == ">" or line == "\n":
                    continue
                line = re.sub("NN+", "", line).replace("\n", "")
                transformation_ADN.cut_and_write_read(f_res, line, k, s)
