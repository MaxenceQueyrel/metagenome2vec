#!/bin/bash

. $METAGENOME2VEC_PATH/metagenome2vec/utils/bash_manager.sh

args $0 "$@"

path_script="$METAGENOME2VEC_PATH/metagenome2vec/read2genome/train_fastdna.py"

if [[ $help == "true" ]]
then
  python $path_script --help
  exit 0
fi

eval $(parse_yaml $conf_file)

name="fastdna_k_${k}_emb_${dim}_epoch_${epoch}_noise_${noise}_${name}"

python $path_script -k $k -E $dim -S $epoch -nc $thread -pd "${input},${labels}" -pm $output -pkv $kmer2vec -f $name -pt $path_tmp_folder -tt $tax_taken -no $noise -R $learning_rate -Ml $max_length
