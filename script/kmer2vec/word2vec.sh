#PBS -S /bin/bash
#PBS -N word2vec
#PBS -l nodes=1:ppn=45
#PBS -q alpha
#PBS -m abe -M maxxxqueyrel@gmail.com
#PBS -o /home/queyrelm/tmp/word2vec.out
#PBS -e /home/queyrelm/tmp/word2vec.err
#PBS -l walltime=24:00:00

conda activate /home/queyrelm/py3-maxence

script="$DEEPGENE/Pipeline/kmer2vec/word2vec_genome.py"
k=${k}
w=${w}
s=${step}
E=${E}
S=${S}
threads=${threads}
R=0.05
path_logs=$DEEPGENE/logs/kmer2vec
path_tmp="/scratchalpha/queyrelm/tmp"
path_data=${path_data}

spark-submit --master local[${threads}] --driver-memory 100g $script -pd $path_data -pa $ANALYSE -E $E -S $S -R $R -w $w -k $k -s $s -ca 506ref -pg $path_logs -pt $path_tmp

