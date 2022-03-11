#!/bin/bash

. $METAGENOME2VEC_PATH/metagenome2vec/utils/bash_manager.sh

args $0 "$@"

path_script="$METAGENOME2VEC_PATH/metagenome2vec/data_processing/simulation/create_camisim_config_file.py"

if [[ $help == "true" ]]
then
  python $path_script --help
  exit 0
fi

eval $(parse_yaml $conf_file)

python $path_script \
    -nc $n_cpus \
    -nsc $n_sample_by_class \
    -ct $computation_type \
    -go $giga_octet \
    -pt $path_tmp_folder \
    -ps $path_save \
    -pap $path_abundance_profile


