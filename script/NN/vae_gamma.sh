#PBS -N vae
#PBS -l nodes=1:ppn=12
#PBS -l ngpus=1
#PBS -l mem=50G
#PBS -q gamma
#PBS -m abe -M maxxxqueyrel@gmail.com
#PBS -o /home/queyrelm/tmp/vae.out
#PBS -e /home/queyrelm/tmp/vae.err
#PBS -l walltime=48:00:00

conda activate /home/queyrelm/py3-metagenome2vec

path_data=${path_data}
path_data=`echo $path_data | tr "#" ","`
path_model=${path_model}
path_metadata=${path_metadata}
path_metadata=`echo $path_metadata | tr "#" ","`
dataset_name=${dataset_name}
disease=${disease}
disease = `echo $disease | tr "#" ","`

batch_size=${batch_size}
if [ -z "$batch_size" ]
then
  batch_size=2
fi
n_steps=${n_steps}
if [ -z "$n_steps" ]
then
  n_steps=50
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
vae_struct=${vae_struct}
if [ -z "$vae_struct" ]
then
  vae_struct="40,1,1"
fi
n_cpus=${n_cpus}
if [ -z "$n_cpus" ]
then
  n_cpus=12
fi
n_iterations=${n_iterations}
if [ -z "$n_iterations" ]
then
  n_iterations=1
fi

id_gpu=${id_gpu}
if [ -z "$id_gpu" ]
then
  id_gpu=-1
fi
id_gpu=`echo $id_gpu | tr "#" ","`
n_memory=${n_memory}
if [ -z "$n_memory" ]
then
  n_memory=8
fi
ressources=${ressources}
if [ -z "$ressources" ]
then
  ressources=cpu:3_gpu:0._worker:5
fi
ressources=`echo $ressources | tr "#" ","`


python $DEEPGENE/Pipeline/NN/vae.py \
  -pd $path_data \
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
  -TU

