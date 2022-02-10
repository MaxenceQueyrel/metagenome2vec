#!/bin/bash

. $METAGENOME2VEC_PATH/metagenome2vec/utils/bash_manager.sh

args $0 "$@"

if [ -z "$conf_file" ]
then
  usage
  exit
fi

eval $(parse_yaml $conf_file)

if [ $overwrite = "True" ]
then
  rm -r $path_save
fi

python $METAGENOME2VEC_PATH/metagenome2vec/data_processing/genome/kmerization.py -pd $path_data -ps $path_save -k $k -s $s

