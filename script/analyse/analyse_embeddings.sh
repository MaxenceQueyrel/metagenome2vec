#PBS -S /bin/bash
#PBS -N analyse_embeddings
#PBS -l nodes=1:ppn=16
#PBS -q alpha 
#PBS -m abe -M maxxxqueyrel@gmail.com
#PBS -o /home/queyrelm/tmp/analyse_embeddings.out
#PBS -e /home/queyrelm/tmp/analyse_embeddings.err
#PBS -l walltime=24:00:00

conda activate $HOME/py3-maxence
pyenv activate py3-maxence

k=$k
ct=$computation_type
pd=$path_data
pkv=$path_kmer2vec
n_cpus=${n_cpus}
if [ -z "$n_cpus" ]
then
  n_cpus=16
fi

python $DEEPGENE/Pipeline/analyse/analyse_embeddings.py -ni 10000 -ct $ct -pd $pd -pkv $pkv -nc $n_cpus

