#PBS -S /bin/bash
#PBS -N analyse_genome_projection
#PBS -l nodes=4:ppn=24
#PBS -q beta
#PBS -m abe -M maxxxqueyrel@gmail.com
#PBS -o /home/queyrelm/tmp/analyse_genome_projection.out
#PBS -e /home/queyrelm/tmp/analyse_genome_projection.err
#PBS -l walltime=24:00:00

data=$DATAB
analyse=$ANALYSEB

export FASTDNA="/scratchbeta/queyrelm/fastDNA"

export SPARK_HOME="/scratchbeta/queyrelm/spark-3.0.1-bin-hadoop2.7"
export MASTER="local[*]"
export SPARK_CONF_DIR="/scratchbeta/queyrelm/spark-3.0.1-bin-hadoop2.7/conf"
export PYSPARK_PYTHON="/home/queyrelm/py3-maxence/bin/python"
export SCRATCHDIR="/scratchbeta/queyrelm"
export PATH=$SPARK_HOME/bin:$PATH
export SPARK_LAUNCHER="pbsdsh"
export SPARK_LOCAL_DIRS="/scratchbeta/queyrelm/tmp"

conda activate /home/queyrelm/py3-maxence
pyenv activate py3-maxence

k=${k}
path_data=${path_data}
path_read2vec=${path_read2vec}
path_save=${path_save}
read2vec=${read2vec}
path_metadata=${path_metadata}

#k=6
#path_data=$DATA/genome/sim_db_concatenated_506/
#path_read2vec=$ANALYSE/read2vec/transformer_k_6.pt
#path_save=$ANALYSE/read2vec/transformer_k_6
#path_metadata=$DEEPGENE/data/506genomes/metadata.csv
#read2vec=transformer
#path_genome_distance=$DEEPGENE/data/taxonomy/matrix_dist.csv


num_executors=${num_executors}
if [ -z "$num_executors" ]
then
  num_executors=3
fi
executor_cores=${executor_cores}
if [ -z "$executor_cores" ]
then
  executor_cores=24
fi
driver_memory=${driver_memory}
if [ -z "$driver_memory" ]
then
  driver_memory="60G"
fi
driver_memory_overhead=${driver_memory_overhead}
if [ -z "$driver_memory_overhead" ]
then
  driver_memory_overhead="60G"
fi
executor_memory=${executor_memory}
if [ -z "$executor_memory" ]
then
  executor_memory="60G"
fi
executor_memory_overhead=${executor_memory_overhead}
if [ -z "$executor_memory_overhead" ]
then
  executor_memory_overhead="60G"
fi
n_cpus=${n_cpus}
if [ -z "$n_cpus" ]
then
  n_cpus=16
fi
path_tmp_folder=${path_tmp_folder}
if [ -z "$path_tmp_folder" ]
then
  path_tmp_folder=/scratchbeta/queyrelm/tmp
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
path_genome_dist=${path_genome_dist}
if [ -z "$num_partitions" ]
then
  path_genome_dist=None
fi


/scratchbeta/queyrelm/pbstools/bin/pbs-spark-submit \
 --no-worker-on-mother-superior \
 --init \
 --ssh \
 --master-interface ib0 \
 --memory 125GB \
 --worker-memory 125GB \
 --driver-memory $driver_memory \
 --conf spark.driver.memoryOverhead=$driver_memory_overhead \
 --executor-memory $executor_memory \
 --conf spark.executor.memoryOverhead=$executor_memory_overhead \
 --conf spark.app.name=genome_projection \
 --conf spark.executor.extraLibraryPath="$DEEPGENE/Pipeline/utils/transformation_ADN.so" \
 --files $DEEPGENE/Pipeline/utils/transformation_ADN.so \
 --conf spark.locality.wait=0 \
 --conf spark.worker.cleanup.enabled=true \
 --conf spark.worker.cleanup.interval=900 \
 --conf spark.sql.autoBroadcastJoinThreshold=-1 \
 --conf spark.scheduler.minRegisteredResourcesRatio=1 \
 --conf spark.network.timeout=800 \
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
 -pgd $path_genome_dist \
 -np $num_partitions
