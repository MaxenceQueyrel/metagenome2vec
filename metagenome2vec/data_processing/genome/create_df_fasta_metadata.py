import pandas as pd
from Bio import SeqIO
import requests
from bs4 import BeautifulSoup
import os
import time
import re
from tqdm import tqdm
import json
from ete3 import NCBITaxa
import numpy as np
import shutil

from metagenome2vec.utils import parser_creator
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
    for (dirpath, dirnames, filenames) in os.walk(path_data):
        L_fasta_file.extend(list(filter(lambda x: x.endswith(".fna") or x.endswith(".fasta"), filenames)))
        break
    return L_fasta_file


def count_bp_length(path_fasta_file):
    """
    Count the base pair lenght of a fasta file
    :param path_fasta_file: String, complete path to the fasta file
    :return: int, bp_length of the fasta file
    """
    with open(path_fasta_file, 'r') as fasta_file:
        total_length = 0
        for cur_record in SeqIO.parse(fasta_file, "fasta"):
            total_length += cur_record.seq.count('A') + \
                            cur_record.seq.count('C') + \
                            cur_record.seq.count('G') + \
                            cur_record.seq.count('T')
    return total_length


def get_taxonomy_id(path_fasta_file):
    """
    Return the taxonomic id of a fasta file by scraping the ncbi website
    :param path_fasta_file: String, complete path to the fasta file
    :return: int, bp_length of the fasta file
    """
    tax_id = None
    with open(path_fasta_file, 'r') as fasta_file:
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
                    page = requests.get('https://www.ncbi.nlm.nih.gov/assembly/{}/'.format(id_))
                    soup = BeautifulSoup(page.content, 'html.parser')
                    tax_link = "https://www.ncbi.nlm.nih.gov/" + re.sub('.*"(.*)".*', '\\1', str(
                        soup.find(id="summary").findAll('a', href=re.compile('^/Taxonomy.*'))[0])).replace("&amp;", "&")
                    cpt = 0
                except:
                    cpt += 1
                    time.sleep(0.5)
                    continue
            # Get the taxonomic id
            try:
                page = requests.get(tax_link)
                soup = BeautifulSoup(page.content, "html.parser")
                tax_id = re.search('Taxonomy ID: ([0-9]*)', str(soup))[1]
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
    return {'{}_id'.format(rank): ranks2lineage.get(rank, '<not present>') for rank in L_rank}


def complete_df_with_all_phylo_id(df, taxa_id="NCBI_ID"):
    """
    Complete a dataframe with several phylogenetic ranks from a column with taxa ids
    :param df: Pandas Dataframe, with one column containing taxa ids
    :param taxa_id: String, name of the column containing taxa ids
    :return:
    """
    L_rank = [kingdom_name, phylum_name, class_name, order_name, family_name, genus_name, species_name, strain_name]
    ncbi = NCBITaxa()
    for i, tax in tqdm(df[taxa_id].iteritems()):
        D_res = get_ranks(int(tax), L_rank, ncbi)
        for rank in L_rank:
            df.loc[i, rank] = int(D_res[rank + "_id"]) if D_res[rank + "_id"] != "<not present>" else -1
        for rank in L_rank:
            df.loc[i, rank + "_name"] = ncbi.translate_to_names([int(D_res[rank + "_id"])])[0] if D_res[rank + "_id"] != "<not present>" else -1
    for rank in L_rank:
        df[rank] = df[rank].apply(int)
    for i in range(2, len(L_rank) + 1):
        j = len(L_rank) - i
        index = df[df[L_rank[j]] == -1].index
        if len(index) > 0:
            df.loc[index, L_rank[j]] = df.loc[index, L_rank[j + 1]]
            df.loc[index, L_rank[j] + "_name"] = df.loc[index, L_rank[j + 1] + "_name"]


def _generate_abundance_name(abundance_name, tax_level):
    """
    give a generic name for the columns of abundance
    :param abundance_name: Str, the name of the abundance
    :param tax_level: Str, the taxonomic level
    :return: Str, generic name for a abundance column
    """
    return abundance_name + "_" + tax_level


def complete_fasta_metadata_with_abundance(df, D_modif_abundance=None):
    """
    Complete a dataframe with abundance columns from bd_length and if not None D_modif_abundance
    :param df: DataFrame Pandas, fasta metadata dataframe
    :param D_modif_abundance: Dictionary like {abundance_name: {"tax_level": tax_level, "modif": {tax_id: factor, ...}} ...}
    :return: DataFrame Pandas, fasta metadata dataframe with abundance
    """
    # Create balanced abundance for ncbi ids
    df[abundance_balanced_name] = 1 / df[bp_length_name]
    df[abundance_balanced_name] /= df[abundance_balanced_name].sum()
    # Create balanced abundance for species and genus level V1
    L_tax_level = [species_name, genus_name]
    for tax_level in L_tax_level:
        abundance_tax_balanced_name = _generate_abundance_name(abundance_balanced_name, tax_level)
        bp_length_name_by_taxa = "%s_by_%s" % (bp_length_name, tax_level)
        gb = df.groupby(tax_level)[bp_length_name].sum().reset_index().rename(columns={bp_length_name: bp_length_name_by_taxa})
        gb[bp_length_name_by_taxa] = 1 / gb[bp_length_name_by_taxa]
        gb[bp_length_name_by_taxa] /= gb[bp_length_name_by_taxa].sum()
        df = df.merge(gb, on=tax_level)
        df[abundance_tax_balanced_name] = 1 / df[bp_length_name]
        df[abundance_tax_balanced_name] = df.groupby(tax_level)[abundance_tax_balanced_name].apply(lambda x: x / x.sum())
        df[abundance_tax_balanced_name] *= df[bp_length_name_by_taxa]
        del df[bp_length_name_by_taxa]
    # Create balanced abundance for species and genus level V2
    bp_length_sum_name = bp_length_name + "_sum"
    bp_length_norm_name = bp_length_name + "_norm"
    for tax_level in L_tax_level:
        abundance_tax_balanced_name = _generate_abundance_name(abundance_balanced_name, tax_level) + "_V2"
        gb = df[[tax_level, bp_length_name]].groupby(tax_level).sum().reset_index()
        gb = gb.rename(columns={bp_length_name: bp_length_sum_name})
        gb[bp_length_norm_name] = gb[bp_length_sum_name] / gb[bp_length_sum_name].sum()
        df = df.merge(gb[[tax_level, bp_length_sum_name, bp_length_norm_name]], on=tax_level)
        df[abundance_tax_balanced_name] = 1 / (df[bp_length_norm_name] * (df[bp_length_name] / df[bp_length_sum_name]))
        df[abundance_tax_balanced_name] /= df.groupby(tax_level)[abundance_tax_balanced_name].transform("sum")
        df[abundance_tax_balanced_name] /= df[abundance_tax_balanced_name].sum()
        del df[bp_length_norm_name]
        del df[bp_length_sum_name]
    # Create balanced abundance for ncbi ids with respect of real uniformity
    df[abundance_balanced_name+"_uniform"] = 1 / df.shape[0]
    # Create modified abundance for a special taxonomic level
    if D_modif_abundance is not None:
        for abundance_modif_name, D_specificities in D_modif_abundance.items():
            tax_level, D_modif = D_specificities[tax_level_name], D_specificities["modif"]
            abundance_modif_name = _generate_abundance_name(abundance_modif_name, tax_level)
            abundance_tax_balanced_name = _generate_abundance_name(abundance_balanced_name, tax_level)
            df[abundance_modif_name] = df[abundance_tax_balanced_name]
            for id_, factor in D_modif.items():
                df.loc[df[tax_level] == int(id_), abundance_modif_name] *= int(factor)
                df[abundance_modif_name] = df[abundance_modif_name] / df[abundance_modif_name].sum()
    return df


def create_df_fasta_metadata(path_fasta_folder):
    """
    Construct the metadata dataframe of a collection of fasta file located in path_data
    :param path_fasta_folder: String, complete path to the folder containing fasta file
    :return: DataFrame Pandas, the complete metadata fatadrame
    """
    L_fasta_file = get_fasta_file(path_fasta_folder)
    df_fasta_metadata = pd.DataFrame(L_fasta_file, columns=[fasta_name])
    df_fasta_metadata[genome_id_name] = ["genome_{}".format(x) for x in range(len(df_fasta_metadata))]
    df_fasta_metadata[novelty_category_name] = known_strain_name
    df_fasta_metadata[OTU_name] = [x for x in range(len(df_fasta_metadata))]
    df_fasta_metadata[ncbi_id_name] = [get_taxonomy_id(os.path.join(path_fasta_folder, fasta_file)) for fasta_file in
                                       tqdm(L_fasta_file)]
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
    df_fasta_metadata[bp_length_name] = [count_bp_length(os.path.join(path_fasta_folder, fasta_file)) for fasta_file in
                                         tqdm(L_fasta_file)]
    complete_df_with_all_phylo_id(df_fasta_metadata)
    df_fasta_metadata = df_fasta_metadata.rename(columns={"strain_name": tax_name})
    return df_fasta_metadata


def create_files_camisim(df_fasta_metadata, path_folder_save, path_fasta_folder, D_modif_abundance=None):
    """
    From the fasta metadata dataset created by the function 'create_df_fasta_metadata', it writes the files to use with camisim.
    :param df_fasta_metadata: DataFrame Pandas, the complete metadata fatadrame with abundance
    :param path_folder_save: Str, Path to the folder where are saved the data
    :param path_fasta_folder: Str, Path to the folder containing fasta file
    :param D_modif_abundance: Dictionary like {abundance_name: {"tax_level": tax_level, "modif": {tax_id: factor, ...}} ...}
    """
    # Create the folder where are save the files for CAMISIM
    os.makedirs(path_folder_save, exist_ok=True)
    # Write genome_to_id.tsv
    df_tmp = df_fasta_metadata[[genome_id_name, fasta_name]].copy()
    df_tmp[fasta_name] = df_tmp[fasta_name].apply(lambda x: os.path.join(os.path.abspath(path_fasta_folder), x))
    df_tmp.to_csv(os.path.join(path_folder_save, "genome_to_id.tsv"), sep="\t", index=False, header=False)
    # Write metadata.tsv, renome genome_id_name to genome_ID for camisim
    df_fasta_metadata[[genome_id_name, OTU_name, ncbi_id_name, novelty_category_name]].rename(columns={genome_id_name: "genome_ID"}, )\
        .to_csv(os.path.join(path_folder_save, "metadata.tsv"), sep="\t", index=False)
    # Write abundance.tsv with balanced abundance
    df_fasta_metadata[[genome_id_name, abundance_balanced_name]].to_csv(
        os.path.join(path_folder_save, abundance_balanced_name + ".tsv"), sep="\t", header=False, index=False)
    df_fasta_metadata[[genome_id_name, abundance_balanced_name+"_uniform"]].to_csv(
        os.path.join(path_folder_save, abundance_balanced_name+"_uniform.tsv"), sep="\t", header=False, index=False)
    # Write abundance.tsv with balanced abundance for a special taxonomic level
    L_tax_level = [species_name, genus_name]
    for tax_level in L_tax_level:
        abundance_name = _generate_abundance_name(abundance_balanced_name, tax_level)
        df_fasta_metadata[[genome_id_name, abundance_name]].to_csv(
            os.path.join(path_folder_save, abundance_name + ".tsv"), sep="\t", header=False, index=False)
    for tax_level in L_tax_level:
        abundance_name = _generate_abundance_name(abundance_balanced_name, tax_level) + "_V2"
        df_fasta_metadata[[genome_id_name, abundance_name]].to_csv(
            os.path.join(path_folder_save, abundance_name + ".tsv"), sep="\t", header=False, index=False)
    # Write abundance.tsv with a modified abundance for a special taxonomic level
    if D_modif_abundance is not None:
        os.makedirs(os.path.join(path_folder_save, "abundance_profile"), exist_ok=True)
        for abundance_name, D_specificities in D_modif_abundance.items():
            tax_level = D_specificities[tax_level_name]
            abundance_name = _generate_abundance_name(abundance_name, tax_level)
            df_fasta_metadata[[genome_id_name, abundance_name]].to_csv(
                os.path.join(path_folder_save, "abundance_profile", abundance_name + ".tsv"), sep="\t", header=False, index=False)


def create_config_file(df_fasta_metadata, path_save, tax_level="species"):
    """
    Function that creates a config dictionary for the simulation expeiments
    :param df_fasta_metadata: DataFrame Pandas, the complete metadata fatadrame with abundance
    :param path_save: String, Name of the json config file
    :param tax_level: Str, the taxonomic level
    :return Dictionary, config dict for abundance
    """
    L_taxa = df_fasta_metadata[tax_level].unique()
    D_res = {}
    for H in [0, 5, 10, 20]:
        for L in [0, 5, 10, 20]:
            if L == 0 and H == 0:
                continue
            A_tax_to_higher = np.random.choice(L_taxa, size=H, replace=False)
            A_tax_to_lower = np.random.choice(list(set(L_taxa).difference(set(A_tax_to_higher))), size=L, replace=False)
            modif = {str(x): 5.0 for x in A_tax_to_higher}
            modif.update({str(x): 0.2 for x in A_tax_to_lower})
            D_res["NT_%s_H_%s_L_%s" % (len(L_taxa), H, L)] = {tax_level_name: tax_level,
                                                              "modif": dict(modif)}
    with open(path_save, "w") as f:
        json.dump(D_res, f)
    return D_res


if __name__ == "__main__":
    args = parser_creator.ParserCreator().parser_create_df_fasta_metadata()
    path_fasta_folder = args.path_data
    path_folder_save = args.path_save
    path_json_modif_abundance = args.path_json_modif_abundance
    path_metadata = args.path_metadata
    simulate_abundance = args.simulate_abundance
    # Execution
    # Create and save fasta metadata
    name_fasta_metadata = os.path.join(path_folder_save, "fasta_metadata.csv")
    os.makedirs(path_folder_save, exist_ok=True)
    if path_metadata is not None:  # Do not compute it if given
        if os.path.exists(name_fasta_metadata):
            os.remove(name_fasta_metadata)
        shutil.copyfile(path_metadata, name_fasta_metadata)
        df_fasta_metadata = pd.read_csv(name_fasta_metadata, sep=",")
    elif os.path.exists(name_fasta_metadata):  # Do not compute it if already exists
        df_fasta_metadata = pd.read_csv(name_fasta_metadata, sep=",")
    else:
        df_fasta_metadata = create_df_fasta_metadata(path_fasta_folder)
        df_fasta_metadata.to_csv(name_fasta_metadata, sep=",", index=False)

    # Complete metadata with abundance and save fasta metadata with abundance
    if path_json_modif_abundance is not None:
        D_modif_abundance = json.load(open(path_json_modif_abundance, 'r'))
    elif simulate_abundance is True:
        D_modif_abundance = create_config_file(df_fasta_metadata, os.path.join(path_folder_save, "config_abundance.json"))
    else:
        D_modif_abundance = None
    name_fasta_metadata_with_abundance = os.path.join(path_folder_save, "fasta_metadata_with_abundance.csv")
    df_fasta_metadata = complete_fasta_metadata_with_abundance(df_fasta_metadata, D_modif_abundance)
    df_fasta_metadata.to_csv(name_fasta_metadata_with_abundance, sep=",", index=False)

    # create and save files for camisim
    path_camisim = os.path.join(path_folder_save, "camisim")
    if os.path.exists(path_camisim):
        shutil.rmtree(path_camisim)
    create_files_camisim(df_fasta_metadata, path_camisim, path_fasta_folder, D_modif_abundance)