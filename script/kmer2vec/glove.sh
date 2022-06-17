#PBS -S /bin/bash
#PBS -N glove
#PBS -l nodes=1:ppn=45
#PBS -q alpha 
#PBS -m abe -M maxxxqueyrel@gmail.com
#PBS -o /home/queyrelm/tmp/glove.out
#PBS -e /home/queyrelm/tmp/glove.err
#PBS -l walltime=24:00:00

conda activate $HOME/py3-maxence

script="$DEEPGENE/Pipeline/kmer2vec/GloVe_genome.py"
k=${k}
w=${w}
s=${step}
X=${X}
E=${E}
S=${S}
path_tmp="/scratchalpha/queyrelm/tmp"
R=0.05
threads=${threads}
path_logs=$DEEPGENE/logs/kmer2vec
path_data=${path_data}

python $script -pd $path_data -pa $ANALYSE -E $E -S $S -R $R -w $w -k $k -s $s -ca 506ref -nc $threads -pg $path_logs -X $X -pt $path_tmp

