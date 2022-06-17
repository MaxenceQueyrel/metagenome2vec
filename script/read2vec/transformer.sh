#PBS -S /bin/bash
#PBS -N transformer
#PBS -l nodes=1:ppn=15
#PBS -l mem=32GB
#PBS -q gpu4
#PBS -m abe -M maxxxqueyrel@gmail.com
#PBS -l walltime=72:00:00
#PBS -e /home/mqueyrel/tmp/transformer.err
#PBS -o /home/mqueyrel/tmp/transformer.out

conda activate /home/queyrelm/py3-maxence
pyenv activate py3-maxence

k=${k}
file_name=${file_name}
path_data_train=${path_data_train}
path_data_valid=${path_data_valid}
n_iteration=${n_iteration}
path_data=$path_data_train,$path_data_valid

n_step=${n_step}
if [ -z "$n_step" ]
then
  n_step=10
fi
batch_size=${batch_size}
if [ -z "$batch_size" ]
then
  batch_size=64
fi
n_cpus=${n_cpus}
if [ -z "$n_cpus" ]
then
  n_cpus=15
fi
embeddings_size=${embeddings_size}
if [ -z "$embeddings_size" ]
then
  embeddings_size=200
fi
hidden_size=${hidden_size}
if [ -z "$hidden_size" ]
then
  hidden_size=200
fi
id_gpu=${id_gpu}
if [ -z "$id_gpu" ]
then
  id_gpu=0
fi
max_length=${max_length}
if [ -z "$max_length" ]
then
  max_length=100
fi
learning_rate=${learning_rate}
if [ -z "$learning_rate" ]
then
  learning_rate=0.001
fi
path_logs=${path_logs}
if [ -z "$path_logs" ]
then
  path_logs=$DEEPGENE/logs/transformer
fi
path_analyse=${path_analyse}
if [ -z "$path_analyse" ]
then
  path_analyse=$ANALYSE
fi
path_kmer2vec=${path_kmer2vec}
if [ -z "$path_kmer2vec" ]
then
  path_kmer2vec=None
fi
nhead=${nhead}
if [ -z "$nhead" ]
then
  nhead=6
fi
nlayers=${nlayers}
if [ -z "$nlayers" ]
then
  nlayers=3
fi
if [ -z "$max_vectors" ]
then
  max_vectors=100000
fi
if [ -z "$path_kmer_count" ]
then
  path_kmer_count=None
fi

python \
 $DEEPGENE/Pipeline/read2vec/transformer.py \
 -k $k \
 -f $file_name \
 -S $n_step \
 -B $batch_size \
 -ig $id_gpu \
 -Ml $max_length \
 -nc $n_cpus \
 -E $embeddings_size \
 -H $hidden_size \
 -R $learning_rate \
 -pg $path_logs \
 -pa $path_analyse \
 -pd $path_data \
 -I $n_iteration \
 -pkv $path_kmer2vec \
 -nl $nlayers \
 -nh $nhead \
 -mv $max_vectors \
 -pkv $path_kmer_count