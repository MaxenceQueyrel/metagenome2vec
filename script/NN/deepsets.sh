#PBS -N deepsets
#PBS -l nodes=1:ppn=50
#PBS -q alpha
#PBS -m abe -M maxxxqueyrel@gmail.com
#PBS -o /home/queyrelm/tmp/deepsets.out
#PBS -e /home/queyrelm/tmp/deepsets.err
#PBS -l walltime=72:00:00

conda activate /home/queyrelm/py3-maxence
pyenv activate py3-maxence

path_data=${path_data}
path_model=${path_model}
path_save=${path_save}
dataset_name=${dataset_name}
n_iterations=${n_iterations}
path_metadata=${path_metadata}

batch_size=${batch_size}
if [ -z "$batch_size" ]
then
  batch_size=12
fi
n_steps=${n_steps}
if [ -z "$n_steps" ]
then
  n_steps=500
fi
learning_rate=${learning_rate}
if [ -z "$learning_rate" ]
then
  learning_rate=0.001
fi
weight_decay=${weight_decay}
if [ -z "$weight_decay" ]
then
  weight_decay=0.1
fi
dropout=${dropout}
if [ -z "$dropout" ]
then
  dropout=0.2
fi
deepsets_struct=${deepsets_struct}
if [ -z "$deepsets_struct" ]
then
  deepsets_struct=300,150,1,1
fi
id_gpu=${id_gpu}
if [ -z "$id_gpu" ]
then
  id_gpu=-1
fi
n_cpus=${n_cpus}
if [ -z "$n_cpus" ]
then
  n_cpus=16
fi
n_memory=${n_memory}
if [ -z "$n_memory" ]
then
  n_memory=8
fi
ressources=${ressources}
if [ -z "$ressources" ]
then
  ressources="cpu:1,gpu:0,worker:5"
fi


python $DEEPGENE/Pipeline/NN/deepSets.py \
  -pd $path_data \
  -pm $path_model \
  -ps $path_save \
  -dn $dataset_name \
  -B $batch_size \
  -S $n_steps \
  -R $learning_rate \
  -D $weight_decay \
  -DO $dropout \
  -DS $deepsets_struct \
  -ig $id_gpu \
  -nm $n_memory \
  -TU \
  -I $n_iterations \
  -pmd $path_metadata \
  -r $ressources



