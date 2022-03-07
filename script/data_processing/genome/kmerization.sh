#!/bin/bash

. $METAGENOME2VEC_PATH/metagenome2vec/utils/bash_manager.sh

args $0 "$@"

path_script="$METAGENOME2VEC_PATH/metagenome2vec/data_processing/simulation/create_camisim_config_file.py"

if $help
then
  python $path_script --help
  exit 0
fi

eval $(parse_yaml $conf_file)

python $path_script -pd $path_data -ps $path_save -k $k -s $s

