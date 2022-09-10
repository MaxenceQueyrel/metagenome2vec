from metagenome2vec.utils import parser_creator

if __name__ == "__main__":
    parserCreator = parser_creator.ParserCreator()
    parser_creator.parser_bok_split()
    parser_creator.parser_bok_merge()
    parser_creator.parser_bok_split()
    parser_creator.parser_bok_merge()
    args = parser_creator.parser.parse_args()
    args = parser_creator.ParserCreator().parser_genome_kmerization()


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