

if __name__ == "__main__":
    args = parser_creator.ParserCreator().parser_create_simulated_read2genome_dataset()
    path_data = args.path_data

    name_matrix_save = reads_genomes_file_name
    name_matrix_save_valid = reads_genomes_file_name + "_valid"
    df_taxonomy_ref = pd.read_csv(args.path_metadata)
    D_taxonomy_mapping = df_taxonomy_ref[[ncbi_id_name, species_name, genus_name, family_name]].astype(str).set_index(ncbi_id_name).to_dict()
    # Create simulation datasets for train and valid
    create_simulated_dataset(path_data, args.valid_size, args.n_sample_load, overwrite=args.overwrite)

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
    if args.valid_size:
        dataframe_to_fastdna_input(path_data, name_matrix_save_valid,
                                   prefix_reads + "_valid",
                                   prefix_tax_id + "_valid",
                                   prefix_species + "_valid",
                                   prefix_genus + "_valid",
                                   prefix_family + "_valid")


