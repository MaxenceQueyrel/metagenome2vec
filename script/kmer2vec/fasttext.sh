#!/bin/bash

. $METAGENOME2VEC_PATH/metagenome2vec/utils/bash_manager.sh

args $0 "$@"

path_script="$METAGENOME2VEC_PATH/metagenome2vec/kmer2vec/fasttext.py"

if [[ $help == "true" ]]
then
  python $path_script --help
  exit 0
fi

eval $(parse_yaml $conf_file)

python $script -pd $path_data -pa $ANALYSE -E $E -S $S -R $R -w $w -k $k -s $step -ca 506ref -nc $threads -pg $path_logs  -pt $path_tmp

