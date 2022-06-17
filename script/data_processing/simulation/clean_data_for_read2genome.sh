#!/bin/bash

. $METAGENOME2VEC_PATH/metagenome2vec/utils/bash_manager.sh

args $0 "$@"

path_script="$METAGENOME2VEC_PATH/metagenome2vec/data_processing/simulation/create_simulated_read2genome_dataset.py"

if [[ $help == "true" ]]
then
  python $path_script --help
  exit 0
fi

eval $(parse_yaml $conf_file)

bash $METAGENOME2VEC_PATH/metagenome2vec/data_processing/simulation/clean_output_simulation.sh $reads_file $reads_mapping_file $path_save $path_metadata

python $path_script -pd $path_save -nsl $n_reads_taken -vs $valid_size -o -pmd $path_metadata
