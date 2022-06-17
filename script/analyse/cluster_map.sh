#PBS -N cluster_map
#PBS -l nodes=1:ppn=1
#PBS -q beta
#PBS -m abe -M maxxxqueyrel@gmail.com
#PBS -o /home/queyrelm/tmp/cluster_map.out
#PBS -e /home/queyrelm/tmp/cluster_map.err
#PBS -l walltime=01:00:00

conda activate /home/queyrelm/py3-maxence
pyenv activate py3-maxence

path_metadata=${path_metadata}
path_data=${path_data}

python $DEEPGENE/Pipeline/analyse/cluster_map.py -pd $path_data -pmd $path_metadata
