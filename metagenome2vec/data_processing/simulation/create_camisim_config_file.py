import os
import sys
import shutil
import numpy as np
root_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.insert(0, os.path.join(root_folder, "utils"))
import parser_creator


if __name__ == "__main__":
    args = parser_creator.ParserCreator().parser_create_camisi_config_file()

    n_cpus = args.n_cpus
    n_sample_by_class = args.n_sample_by_class
    computation_type = args.computation_type
    size = args.giga_octet
    path_tmp_folder = os.path.abspath(args.path_tmp_folder)
    path_save = os.path.abspath(args.path_save)
    path_abundance_profile = os.path.abspath(args.path_abundance_profile)
    name_config_folder = "config_files"
    name_camisim_folder = "camisim"
    name_abundance_profile = "abundance_profile"
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
    n_genomes = sum(1 for line in open(os.path.join(path_save, name_camisim_folder, file_name_genome_to_id)))

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
