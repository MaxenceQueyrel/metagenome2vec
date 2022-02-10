#!/bin/bash

. $METAGENOME2VEC_PATH/metagenome2vec/utils/bash_manager.sh

args $0 "$@"

if [ -z "$conf_file" ]
then
  usage
  exit
fi

eval $(parse_yaml $conf_file)


python $METAGENOME2VEC_PATH/metagenome2vec/data_processing/simulation/create_camisim_config_file.py \
    -nc $n_cpus \
    -nsc $n_sample_by_class \
    -ct $computation_type \
    -go $giga_octet \
    -pt $path_tmp_folder \
    -ps $path_save \
    -pap $path_abundance_profile


