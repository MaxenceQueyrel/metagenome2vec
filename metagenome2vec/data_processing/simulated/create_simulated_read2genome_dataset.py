import os
import sys

root_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.insert(0, os.path.join(root_folder, "utils"))

from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import re
import parser_creator
from string_names import *
SEED = 42


def dataframe_to_fastdna_input(path_data, name_matrix, name_reads_fastdna, name_ids_fastdna,
                               name_species_fastdna, name_genus_fastdna, name_family_fastdna):
    name_matrix = os.path.join(path_data, name_matrix)
    name_reads_fastdna = os.path.join(path_data, name_reads_fastdna)
    name_ids_fastdna = os.path.join(path_data, name_ids_fastdna)
    name_species_fastdna = os.path.join(path_data, name_species_fastdna)
    name_genus_fastdna = os.path.join(path_data, name_genus_fastdna)
    name_family_fastdna = os.path.join(path_data, name_family_fastdna)
    with open(name_matrix, "r") as f:
        with open(name_reads_fastdna, "w") as out_reads:
            with open(name_ids_fastdna, "w") as out_tax_id:
                with open(name_species_fastdna, "w") as out_species:
                    with open(name_genus_fastdna, "w") as out_genus:
                        with open(name_family_fastdna, "w") as out_family:
                            for i, line in tqdm(enumerate(f)):
                                if i == 0:
                                    continue
                                read, tax_id, sim_id, ratio = line.split('\t')
                                out_reads.write(">%s\n" % str(i))
                                out_reads.write(read + "\n")
                                out_tax_id.write(tax_id + "\n")
                                out_genus.write(D_taxonomy_mapping[genus_name][tax_id] + "\n")
                                out_family.write(D_taxonomy_mapping[family_name][tax_id] + "\n")
                                out_species.write(D_taxonomy_mapping[species_name][tax_id] + "\n")


def create_simulated_dataset(path_data, valid_size=None, n_sample_load=-1, overwrite=False):
    """
    Load or create the simulated data matrix
    :param path_data: String, Path to folder, assuming that it exists a file named reads and a file
    named mapping_read_genome
    :param valid_size, float, percentage for the validation datase
    :param n_sample_load, long, number of read loaded
    :param overwrite: bool, if true replace matrix if exists
    :return:
    """
    path_final_matrix = os.path.join(path_data, name_matrix_save)
    path_final_matrix_valid = os.path.join(path_data, name_matrix_save_valid)
    if overwrite or not os.path.exists(path_final_matrix):
        path_mapping_read_genome = os.path.join(path_data, mapping_read_file_name)
        path_reads = os.path.join(path_data, reads_file_name)
        mapping_read_genome = pd.read_csv(path_mapping_read_genome, sep="\t", dtype=str)[[tax_id_name, anonymous_read_id_name]]
        reads = pd.read_csv(path_reads, sep="\t")  # 1 min
        if n_sample_load > 0:
            reads = reads.sample(n=n_sample_load, random_state=SEED)
        reads[anonymous_read_id_name] = reads[anonymous_read_id_name].apply(lambda x: re.sub("(.*)-.*$", "\\1", x.replace("@", "")))
        reads = reads.merge(mapping_read_genome, on=anonymous_read_id_name)[[read_name, tax_id_name, sim_id_name]]

        def compute_proportion(df):
            sim_ids = df[sim_id_name].unique()
            df[prop_name] = np.zeros(df.shape[0], dtype=np.float)
            for sim_id in tqdm(sim_ids):
                tmp = df[df[sim_id_name] == sim_id]
                tmp = tmp.groupby(tax_id_name).transform("count")[prop_name] * 1. / tmp.shape[0]
                df.loc[tmp.index, prop_name] = tmp
            return df

        if valid_size > 0:
            try:
                reads, reads_valid = train_test_split(reads, test_size=valid_size, random_state=SEED, stratify=reads[tax_id_name])
            except:
                # Try to stratify at the species level if the stratification at the genome level didn't work
                reads = reads.merge(df_taxonomy_ref[[ncbi_id_name, species_name]].astype(str), left_on=tax_id_name, right_on=ncbi_id_name)
                reads, reads_valid = train_test_split(reads, test_size=valid_size, random_state=SEED, stratify=reads[species_name])
                reads = reads.drop([species_name, ncbi_id_name], axis=1)
                reads_valid = reads_valid.drop([species_name, ncbi_id_name], axis=1)
            reads_valid = compute_proportion(reads_valid)
            reads_valid.to_csv(path_final_matrix_valid, index=False, sep="\t")
        reads = compute_proportion(reads)
        reads.to_csv(path_final_matrix, index=False, sep="\t")


if __name__ == "__main__":
    args = parser_creator.ParserCreator().parser_create_simulated_read2genome_dataset()
    n_sample_load = args.n_sample_load
    overwrite = args.overwrite
    path_data = args.path_data
    valid_size = args.valid_size
    path_metadata = args.path_metadata

    name_matrix_save = reads_genomes_file_name
    name_matrix_save_valid = reads_genomes_file_name + "_valid"
    df_taxonomy_ref = pd.read_csv(path_metadata)
    D_taxonomy_mapping = df_taxonomy_ref[[ncbi_id_name, species_name, genus_name, family_name]].astype(str).set_index(ncbi_id_name).to_dict()
    # Create simulated datasets for train and valid
    create_simulated_dataset(path_data, valid_size, n_sample_load, overwrite=overwrite)

    # Create fastdna inputs
    prefix_reads = "in_fastdna_reads"
    prefix_tax_id = "in_fastdna_tax_id"
    prefix_species = "in_fastdna_species"
    prefix_genus = "in_fastdna_genus"
    prefix_family = "in_fastdna_family"
    dataframe_to_fastdna_input(path_data, name_matrix_save,
                               prefix_reads,
                               prefix_tax_id,
                               prefix_species,
                               prefix_genus,
                               prefix_family)
    if valid_size is not None:
        dataframe_to_fastdna_input(path_data, name_matrix_save_valid,
                                   prefix_reads + "_valid",
                                   prefix_tax_id + "_valid",
                                   prefix_species + "_valid",
                                   prefix_genus + "_valid",
                                   prefix_family + "_valid")


