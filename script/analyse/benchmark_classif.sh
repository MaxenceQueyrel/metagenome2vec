#PBS -N benchmark_classif
#PBS -l nodes=1:ppn=50
#PBS -q alpha
#PBS -m abe -M maxxxqueyrel@gmail.com
#PBS -o /home/queyrelm/tmp/benchmark_sil.out
#PBS -e /home/queyrelm/tmp/benchmark_sil.err
#PBS -l walltime=5:00:00

conda activate /home/queyrelm/py3-maxence
pyenv activate py3-maxence

path_data=${path_data}
path_save=${path_save}
n_iter=${n_iter}
dataset_name=${dataset_name}
n_cpus=${n_cpus}
path_metadata=${path_metadata}
is_bok=${is_bok}

if [ -z "$is_bok" ]
then
  python $DEEPGENE/Pipeline/analyse/benchmark_classif.py -pd $path_data -ps $path_save -nc $n_cpus -I $n_iter -dn $dataset_name -pmd $path_metadata
else
  python $DEEPGENE/Pipeline/analyse/benchmark_classif.py -pd $path_data -ps $path_save -nc $n_cpus -I $n_iter -dn $dataset_name -pmd $path_metadata -ib
fi


