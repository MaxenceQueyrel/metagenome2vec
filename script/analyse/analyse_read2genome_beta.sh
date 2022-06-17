#PBS -N analyse_read2genome
#PBS -l nodes=3:ppn=24
#PBS -l mem=124GB
#PBS -q beta
#PBS -o /home/queyrelm/tmp/analyse_read2genome.out
#PBS -e /home/queyrelm/tmp/analyse_read2genome.err
#PBS -l walltime=10:00:00

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

path_model=${path_model}
path_data_train=${path_data_train}
path_data_valid=${path_data_valid}
path_save=${path_save}
read2genome=${read2genome}
path_metadata=${path_metadata}

driver_memory=${driver_memory}
if [ -z "$driver_memory" ]
then
  driver_memory="32g"
fi
driver_memory_overhead=${driver_memory_overhead}
if [ -z "$driver_memory_overhead" ]
then
  driver_memory_overhead="32g"
fi
executor_memory=${executor_memory}
if [ -z "$executor_memory" ]
then
  executor_memory="32g"
fi
executor_memory_overhead=${executor_memory_overhead}
if [ -z "$executor_memory_overhead" ]
then
  executor_memory_overhead="32g"
fi
tax_level=${tax_level}
if [ -z "$tax_level" ]
then
  tax_level=species
fi
n_sample_load=${n_sample_load}
if [ -z "$n_sample_load" ]
then
  n_sample_load=-1
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
tax_taken=${tax_taken}
if [ -z "$tax_taken" ]
then
  tax_taken=None
fi

/scratchbeta/queyrelm/pbstools/bin/pbs-spark-submit \
 --init \
 --ssh \
 --master-interface ib0 \
 --driver-memory $driver_memory \
 --conf spark.driver.memoryOverhead=$driver_memory_overhead \
 --executor-memory $executor_memory \
 --conf spark.executor.memoryOverhead=$executor_memory_overhead \
 --conf spark.app.name=read2genome \
 --conf spark.locality.wait=0 \
 --conf spark.sql.autoBroadcastJoinThreshold=-1 \
 --conf spark.scheduler.minRegisteredResourcesRatio=1 \
 --conf spark.executor.extraLibraryPath="$DEEPGENE/Pipeline/utils/transformation_ADN.so" \
 --conf spark.cleaner.referenceTracking=false \
 --conf spark.cleaner.referenceTracking.blocking=false \
 --conf spark.cleaner.referenceTracking.blocking.shuffle=false \
 --conf spark.cleaner.referenceTracking.cleanCheckpoints=false \
 --conf spark.network.timeout=800 \
 --conf spark.sql.execution.arrow.maxRecordsPerBatch=50000 \
 --files $DEEPGENE/Pipeline/utils/transformation_ADN.so \
 $DEEPGENE/Pipeline/analyse/analyse_read2genome.py \
 -pd $path_data_train,$path_data_valid \
 -pg $DEEPGENE/logs/analyse_read2genome \
 -mo local \
 -tl $tax_level \
 -pm $path_model \
 -nsl $n_sample_load \
 -nc $n_cpus \
 -rg $read2genome \
 -pt $path_tmp_folder \
 -tt $tax_taken \
 -pmd $path_metadata

