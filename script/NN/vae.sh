#!/bin/bash

. $METAGENOME2VEC_PATH/metagenome2vec/utils/bash_manager.sh

args $0 "$@"

path_script="$METAGENOME2VEC_PATH/metagenome2vec/NN/vae.py"

if [[ $help == "true" ]]
then
  python $path_script --help
  exit 0
fi

eval $(parse_yaml $conf_file)

string_args="-pd $path_data \
  -pm $path_model \
  -d $disease \
  -dn $dataset_name \
  -B $batch_size \
  -S $n_steps \
  -R $learning_rate \
  -DO $dropout \
  -DV $vae_struct \
  -ig $id_gpu \
  -nm $n_memory \
  -I $n_iterations \
  -pmd $path_metadata \
  -D $weight_decay \
  -r $ressources \
  -cv $cross_validation \
  -AF $activation_function"


if [[ $tune == "True" ]]; then
  string_args=$string_args" -TU"
fi

python $path_script $string_args
  

