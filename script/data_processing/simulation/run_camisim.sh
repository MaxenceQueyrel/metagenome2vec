#!/bin/bash

. $METAGENOME2VEC_PATH/metagenome2vec/utils/bash_manager.sh

args $0 "$@"

if [ -z "$conf_file" ]
then
  usage
  exit
fi

eval $(parse_yaml $conf_file)

python $CAMISIM/metagenomesimulation.py --debug $init_file


