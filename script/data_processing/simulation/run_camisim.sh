#!/bin/bash

. $METAGENOME2VEC_PATH/metagenome2vec/utils/bash_manager.sh

args $0 "$@"

path_script="$CAMISIM/metagenomesimulation.py"

if $help
then
  python $path_script --help
  exit 0
fi

eval $(parse_yaml $conf_file)

python $path_script --debug $init_file


