#PBS -S /bin/bash
#PBS -N analyse_genome_projection
#PBS -l nodes=1:ppn=15
#PBS -l mem=32GB
#PBS -q ican
#PBS -m abe -M maxxxqueyrel@gmail.com
#PBS -l walltime=20:00:00
#PBS -e /home/mqueyrel/tmp/analyse_genome_projection.err
#PBS -o /home/mqueyrel/tmp/analyse_genome_projection.out

conda activate /home/queyrelm/py3-maxence
pyenv activate py3-maxence

k=${k}
path_data=${path_data}
path_read2vec=${path_read2vec}
path_save=${path_save}
read2vec=${read2vec}
path_metadata=$DEEPGENE/data/506genomes/metadata.csv
path_genome_distance=$DEEPGENE/data/506genomes/matrix_dist.csv

#k=6
#path_data=$DATA/genome/sim_db_concatenated_506/
#path_read2vec=$ANALYSE/read2vec/transformer_k_6.pt
#path_save=$ANALYSE/read2vec/transformer_k_6
#path_metadata=$DEEPGENE/data/506genomes/metadata.csv
#read2vec=transformer
#path_genome_distance=$DEEPGENE/data/506genomes/matrix_dist.csv


driver_memory=${driver_memory}
if [ -z "$driver_memory" ]
then
  driver_memory=50g
fi
driver_memory_overhead=${driver_memory_overhead}
if [ -z "$driver_memory_overhead" ]
then
  driver_memory_overhead=60g
fi
n_cpus=${n_cpus}
if [ -z "$n_cpus" ]
then
  n_cpus=16
fi
path_tmp_folder=${path_tmp_folder}
if [ -z "$path_tmp_folder" ]
then
  path_tmp_folder="~/"
fi
max_length=${max_length}
if [ -z "$max_length" ]
then
  max_length=100
fi
id_gpu=${id_gpu}
if [ -z "$id_gpu" ]
then
  id_gpu=-1
fi
path_metagenome_word_count=${path_metagenome_word_count}
if [ -z "$path_metagenome_word_count" ]
then
  path_metagenome_word_count=None
fi
num_partitions=${num_partitions}
if [ -z "$num_partitions" ]
then
  num_partitions=50
fi


spark-submit \
 --conf spark.app.name=read2genome \
 --conf spark.locality.wait=0 \
 --conf spark.sql.autoBroadcastJoinThreshold=-1 \
 --conf spark.scheduler.minRegisteredResourcesRatio=1 \
 --conf spark.executor.extraLibraryPath="$DEEPGENE/Pipeline/utils/transformation_ADN.so" \
 --conf spark.cleaner.referenceTracking=false \
 --conf spark.cleaner.referenceTracking.blocking=false \
 --conf spark.cleaner.referenceTracking.blocking.shuffle=false \
 --conf spark.cleaner.referenceTracking.cleanCheckpoints=false \
 --master local[*] \
 --driver-memory $driver_memory \
 --conf spark.network.timeout=800 \
 --conf spark.driver.memoryOverhead=$driver_memory_overhead \
 --conf spark.local.dir=$TMP \
 --py-files $DEEPGENE/Pipeline/utils/hdfs_functions.py,\
$DEEPGENE/Pipeline/utils/parser_creator.py,\
$DEEPGENE/Pipeline/read2vec/read2vec.py,\
$DEEPGENE/Pipeline/read2vec/basic.py,\
$DEEPGENE/Pipeline/read2vec/seq2seq.py,\
$DEEPGENE/Pipeline/read2vec/SIF.py,\
$DEEPGENE/Pipeline/utils/transformation_ADN2.py,\
$DEEPGENE/Pipeline/utils/data_manager.py\
 --files $DEEPGENE/Pipeline/utils/transformation_ADN.so \
 $DEEPGENE/Pipeline/analyse/analyse_genome_projection.py \
 -k $k \
 -pd $path_data \
 -ps $path_save \
 -prv $path_read2vec \
 -pg $DEEPGENE/logs/genome_projection \
 -rv $read2vec \
 -nc $n_cpus \
 -pmwc $path_metagenome_word_count \
 -pt $path_tmp_folder \
 -ig $id_gpu \
 -pmd $path_metadata \
 -Ml $max_length \
 -pgd $path_genome_distance \
 -np $num_partitions

