import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import re
import shutil
import subprocess
import csv
import json

from metagenome2vec.utils.string_names import *
from metagenome2vec.data_processing import genome

SEED = 42

# from create_simulated_read2genome_dataset
def dataframe_to_fastdna_input(path_data, path_metadata, name_reads_fastdna="reads_fastdna", name_ids_fastdna="ids_fastdna",
name_species_fastdna="species_fastdna", name_genus_fastdna="genus_fastdna", name_family_fastdna="family_fastdna"):
    df_taxonomy_ref = pd.read_csv(path_metadata)
    D_taxonomy_mapping = df_taxonomy_ref[[ncbi_id_name, species_name, genus_name, family_name]].astype(str).set_index(ncbi_id_name).to_dict()
    path_save = os.path.dirname(path_data)
    name_reads_fastdna = os.path.join(path_save, name_reads_fastdna)
    name_ids_fastdna = os.path.join(path_save, name_ids_fastdna)
    name_species_fastdna = os.path.join(path_save, name_species_fastdna)
    name_genus_fastdna = os.path.join(path_save, name_genus_fastdna)
    name_family_fastdna = os.path.join(path_save, name_family_fastdna)
    with open(path_data, "r") as f:
        with open(name_reads_fastdna, "w") as out_reads:
            with open(name_ids_fastdna, "w") as out_tax_id:
                with open(name_species_fastdna, "w") as out_species:
                    with open(name_genus_fastdna, "w") as out_genus:
                        with open(name_family_fastdna, "w") as out_family:
                            for i, line in tqdm(enumerate(f)):
                                if i == 0:
                                    continue
                                read, tax_id, _, _ = line.split('\t')
                                out_reads.write(">%s\n" % str(i))
                                out_reads.write(read + "\n")
                                out_tax_id.write(tax_id + "\n")
                                out_genus.write(D_taxonomy_mapping[genus_name][tax_id] + "\n")
                                out_family.write(D_taxonomy_mapping[family_name][tax_id] + "\n")
                                out_species.write(D_taxonomy_mapping[species_name][tax_id] + "\n")


# from create_simulated_read2genome_dataset TODO Add the clean output simulation in it
def create_simulated_read2genome_dataset(path_fastq_file, path_mapping_file, path_metadata, path_save,
                                        valid_size=0.3, n_sample_load=-1, overwrite=False):
    """
    # TODO
    """
    assert valid_size >= 0 and valid_size <= 1, "Valid size should be between 0 and 1."
    
    # Clean output simulation if does not exist
    path_mapping_read_genome = os.path.join(path_save, mapping_read_file_name)
    path_reads = os.path.join(path_save, reads_file_name)
    if not ( os.path.exists(path_mapping_read_genome) and os.path.exists(path_reads) ):
        clean_output_simulation(path_fastq_file, path_mapping_file, path_save, path_metadata)
    
    # Create the read2genome datasets
    if overwrite or not os.path.exists(path_final_matrix):

        name_matrix_save = "reads_read2genome"
        name_matrix_save_valid = name_matrix_save + "_valid"
        path_final_matrix = os.path.join(path_save, name_matrix_save)
        path_final_matrix_valid = os.path.join(path_save, name_matrix_save_valid)

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

        try:
            reads, reads_valid = train_test_split(reads, test_size=valid_size, random_state=SEED, stratify=reads[tax_id_name])
        except:
            # Try to stratify at the species level if the stratification at the genome level didn't work
            df_taxonomy_ref = pd.read_csv(path_metadata).astype(str)
            reads = reads.merge(df_taxonomy_ref[[ncbi_id_name, species_name]].astype(str), left_on=tax_id_name, right_on=ncbi_id_name)
            reads, reads_valid = train_test_split(reads, test_size=valid_size, random_state=SEED, stratify=reads[species_name])
            reads = reads.drop([species_name, ncbi_id_name], axis=1)
            reads_valid = reads_valid.drop([species_name, ncbi_id_name], axis=1)
        reads_valid = compute_proportion(reads_valid)
        reads_valid.to_csv(path_final_matrix_valid, index=False, sep="\t")
        reads = compute_proportion(reads)
        reads.to_csv(path_final_matrix, index=False, sep="\t")

        # create files in fastdna format
        dataframe_to_fastdna_input(path_final_matrix, path_metadata)
        dataframe_to_fastdna_input(path_final_matrix_valid, path_metadata, name_reads_fastdna="reads_fastdna_valid", name_ids_fastdna="ids_fastdna_valid",
                        name_species_fastdna="species_fastdna_valid", name_genus_fastdna="genus_fastdna_valid", name_family_fastdna="family_fastdna_valid")


def create_simulated_metagenome2vec_dataset(path_data: str, path_save: str, overwrite: bool = False, to_merge: bool = False):
    if overwrite and os.path.exists(path_save):
        shutil.rmtree(path_save)
    os.makedirs(path_save, exist_ok=True)
    csvfile = open(os.path.join(path_save, 'metadata.csv'), 'w', newline='')
    writter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writter.writerow(["id.fasta", "group", "id.subject"])
    cpt = 0

    merge_file_name = "metagenome.fq.gz"
    for path in path_data.split(","):
        for root, _, files in os.walk(path):
            if os.path.basename(root) == "reads":
                class_name = root.split("/")[-3]
                sample_name = "sample_%s" % cpt
                dir_dst = os.path.join(path_save, sample_name)
                os.makedirs(dir_dst, exist_ok=True)
                if to_merge:
                    if os.path.isfile(os.path.join(dir_dst, merge_file_name)):
                        subprocess.call("rm %s" % os.path.join(dir_dst, merge_file_name), shell=True)
                    subprocess.call("cat %s/*.fq.gz > %s" % (root, os.path.join(dir_dst, merge_file_name)),
                                    shell=True)
                    writter.writerow([sample_name, class_name, sample_name])
                    cpt += 1
                else:
                    for file in files:
                        if "fq.gz" in file:
                            if os.path.exists(os.path.join(dir_dst, file)):
                                os.remove(os.path.join(dir_dst, file))
                            shutil.copy(os.path.join(root, file), os.path.join(dir_dst, file))
                            writter.writerow([sample_name, class_name, sample_name])
                            cpt += 1
    csvfile.close()


def create_simulated_config_file(n_cpus, n_sample_by_class, computation_type, size,
                                 path_tmp_folder, path_save, path_abundance_profile):
    name_config_folder = "config_files"
    name_camisim_folder = "camisim"
    path_save_config = os.path.abspath(os.path.join(path_save, name_camisim_folder, name_config_folder))

    file_name_metadata = "metadata.tsv"
    file_name_genome_to_id = "genome_to_id.tsv"

    try:
        path_camisim = os.path.abspath(os.environ["CAMISIM"])
    except KeyError as error:
        raise KeyError("You must define the global variables: CAMISIM")

    if computation_type == "both" or computation_type == "nanosim":
        try:
            path_nanosim = os.path.abspath(os.environ["NANOSIM"])
        except KeyError as error:
            raise KeyError("You must define the global variables: NANOSIM")

    if os.path.isdir(path_abundance_profile):
        L_file_abundance = os.listdir(path_abundance_profile)
    else:
        L_file_abundance = []
    os.makedirs(path_save_config, exist_ok=True)
    n_genomes = sum(1 for _ in open(os.path.join(path_save, name_camisim_folder, file_name_genome_to_id)))

    L_computation_type = ["illumina"]
    if computation_type == "both":
        L_computation_type = ["illumina", "nanosim"]
    elif computation_type == "nanosim":
        L_computation_type = ["nanosim"]
    for computation_type in L_computation_type:
        name_abundance = path_abundance_profile.split("/")[-1]
        if not L_file_abundance:  # Change name_abundance if path_abundance_profile is a file
            name_abundance = name_abundance.rsplit(".", 1)[0]
        with open(os.path.join(path_save_config, "%s_%s.ini" % (computation_type, name_abundance)), "w") as f_res:
            name_simulation = computation_type + "_" + name_abundance + "_" + str(n_sample_by_class)
            path_save_simulation = os.path.abspath(os.path.join(path_save, name_camisim_folder, "dataset", name_simulation))
            if os.path.exists(path_save_simulation):
                shutil.rmtree(path_save_simulation)
            os.makedirs(path_save_simulation)
            s = "[Main]\n"
            s += "seed=%s\n" % np.random.randint(0, 32741178)
            s += "phase=0\n"
            s += "max_processors=%s\n" % n_cpus
            s += "dataset_id={}\n".format(name_simulation)
            s += "output_directory=%s\n" % path_save_simulation
            s += "temp_directory=%s\n" % path_tmp_folder
            s += "gsa=True\n"
            s += "pooled_gsa=True\n"
            s += "anonymous=True\n"
            s += "compress=1\n"
            s += "\n"
            s += "[ReadSimulator]\n"
            readsim = os.path.join(path_nanosim, 'src/simulator.py') if computation_type == "nanosim" else os.path.join(path_camisim, "tools/art_illumina-2.3.6/art_illumina")
            s += "readsim=%s\n" % readsim
            error = os.path.join(path_camisim, "tools/nanosim_profile/") if computation_type == "nanosim" else os.path.join(path_camisim, "tools/art_illumina-2.3.6/profiles")
            s += "error_profiles=%s\n" % error
            s += "samtools=%s\n" % os.path.join(path_camisim, "tools/samtools-1.3/samtools")
            s += "profil=mbarc\n"
            s += "size=%s\n" % size
            type = "nanosim" if computation_type == "nanosim" else "art"
            s += "type=%s\n" % type
            fragments_size_mean = "" if computation_type == "nanosim" else "270"
            s += "fragments_size_mean=%s\n" % fragments_size_mean
            fragment_size_standard_deviation = "" if computation_type == "nanosim" else "27"
            s += "fragment_size_standard_deviation=%s\n" % fragment_size_standard_deviation
            s += "\n"
            s += "[CommunityDesign]\n"
            s += "distribution_file_paths=%s\n" % (",".join([path_abundance_profile] * n_sample_by_class) if not L_file_abundance else ",".join([os.path.join(path_abundance_profile, file_abundance) for file_abundance in L_file_abundance]))
            s += "ncbi_taxdump=%s\n" % os.path.join(path_camisim, "tools/ncbi-taxonomy_20170222.tar.gz")
            s += "strain_simulation_template=%s\n" % os.path.join(path_camisim, "scripts/StrainSimulationWrapper/sgEvolver/simulation_dir")
            s += "number_of_samples=%s\n" % (n_sample_by_class if not L_file_abundance else len(L_file_abundance))
            s += "\n"
            s += "[community0]\n"
            s += "metadata=%s\n" % os.path.join(path_save, name_camisim_folder, file_name_metadata)
            s += "id_to_genome_file=%s\n" % os.path.join(path_save, name_camisim_folder, file_name_genome_to_id)
            s += "id_to_gff_file=\n"
            s += "genomes_total=%s\n" % n_genomes
            s += "genomes_real=%s\n" % n_genomes
            s += "max_strains_per_otu=1\n"
            s += "ratio=1\n"
            s += "mode=differential\n"
            s += "log_mu=1\n"
            s += "log_sigma=2\n"
            s += "gauss_mu=1\n"
            s += "gauss_sigma=1\n"
            s += "view=False\n\n"
            f_res.write(s)


def randomly_alterate_abundance(df_fasta_metadata, path_save, tax_level="species"):
    """
    Function that creates a config dictionary for the simulation experiments
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


def clean_output_simulation(path_fastq_file: str, path_mapping_file: str, path_save_folder: str, path_metadata: str):
    """_summary_

    Args:
        path_fastq_file (str): _description_
        path_mapping_file (str): _description_
        path_save_folder (str): _description_
        path_metadata (str): _description_
    """
    path_script: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clean_output_simulation.sh")
    subprocess.call([path_script, path_fastq_file, path_mapping_file, path_save_folder, path_metadata])


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


def create_df_fasta_metadata(path_fasta_folder: str, path_folder_save: str, path_json_modif_abundance: str = None, path_metadata: str = None, simulate_abundance: bool = False):
    """_summary_

    Args:
        path_fasta_folder (str): _description_
        path_folder_save (str): _description_
        path_json_modif_abundance (str): _description_
        path_metadata (str): _description_
        simulate_abundance (bool): _description_
    """
    # Create and save fasta metadata
    name_fasta_metadata = os.path.join(path_folder_save, "fasta_metadata.csv")
    os.makedirs(path_folder_save, exist_ok=True)
    if path_metadata is not None:
        if os.path.exists(name_fasta_metadata):
            os.remove(name_fasta_metadata)
        shutil.copyfile(path_metadata, name_fasta_metadata)
        df_fasta_metadata = pd.read_csv(name_fasta_metadata, sep=",")
    elif os.path.exists(name_fasta_metadata):  # Do not compute it if already exists
        df_fasta_metadata = pd.read_csv(name_fasta_metadata, sep=",")
    else:
        df_fasta_metadata = genome.create_df_fasta_metadata(path_fasta_folder)
        df_fasta_metadata.to_csv(name_fasta_metadata, sep=",", index=False)

    # Complete metadata with abundance and save fasta metadata with abundance
    if path_json_modif_abundance is not None:
        D_modif_abundance = json.load(open(path_json_modif_abundance, 'r'))
    elif simulate_abundance is True:
        D_modif_abundance = randomly_alterate_abundance(df_fasta_metadata, os.path.join(path_folder_save, "config_abundance.json"))
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

