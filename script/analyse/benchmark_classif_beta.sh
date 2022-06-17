#PBS -S /bin/bash
#PBS -N benchmark_classif
#PBS -l nodes=1:ppn=24
#PBS -q beta
#PBS -o /home/queyrelm/tmp/benchmark_classif.out
#PBS -e /home/queyrelm/tmp/benchmark_classif.err
#PBS -l walltime=5:00:00

data=$DATAB
analyse=$ANALYSEB

conda activate /home/queyrelm/py3-maxence

export TMPDIR="/scratchbeta/queyrelm/tmp"

path_data=${path_data}
path_save=${path_save}
dataset_name=${dataset_name}
path_metadata=${path_metadata}
is_bok=${is_bok}
disease = ${disease}

n_iter=${n_iter}
if [ -z "$n_iter" ]
then
  n_iter=100
fi
n_cpus=${n_cpus}
if [ -z "$n_cpus" ]
then
  n_cpus=24
fi
test_size=${test_size}
if [ -z "$test_size" ]
then
  test_size=0.2
fi
n_splits=${n_splits}
if [ -z "$n_splits" ]
then
  n_splits=20
fi


if [ -z "$is_bok" ]
then
  python $DEEPGENE/Pipeline/analyse/benchmark_classif.py -pd $path_data -ps $path_save -nc $n_cpus -I $n_iter -dn $dataset_name -pmd $path_metadata -TS $test_size -cv $n_splits -d $disease
else
  python $DEEPGENE/Pipeline/analyse/benchmark_classif.py -pd $path_data -ps $path_save -nc $n_cpus -I $n_iter -dn $dataset_name -pmd $path_metadata -TS $test_size -cv $n_splits -d $disease -ib
fi


