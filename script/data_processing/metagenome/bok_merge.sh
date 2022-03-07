#!/bin/bash

. $METAGENOME2VEC_PATH/metagenome2vec/utils/bash_manager.sh

args $0 "$@"

path_script="$METAGENOME2VEC_PATH/metagenome2vec/data_processing/metagenome/bok_merge.py"

if $help
then
  python $path_script --help
  exit 0
fi

eval $(parse_yaml $conf_file)

if [ $overwrite = "True" ]
then
  rm -r $path_data/k_"$k"_s_"$s"/bok.parquet
fi

while [ ! -d $path_data/k_"$k"_s_"$s"/bok.parquet ]
do
  spark-submit \
   --conf spark.app.name=bok_merge \
   --conf spark.locality.wait=0 \
   --conf spark.sql.autoBroadcastJoinThreshold=-1 \
   --conf spark.scheduler.minRegisteredResourcesRatio=1 \
   --conf spark.executor.extraLibraryPath="$METAGENOME2VEC_PATH/metagenome2vec/utils/transformation_ADN.cpython-38-x86_64-linux-gnu.so" \
   --conf spark.cleaner.referenceTracking=false \
   --conf spark.cleaner.referenceTracking.blocking=false \
   --conf spark.cleaner.referenceTracking.blocking.shuffle=false \
   --conf spark.cleaner.referenceTracking.cleanCheckpoints=false \
   --conf spark.rpc.message.maxSize=1024 \
   --num-executors $num_executors \
   --executor-cores $executor_cores \
   --driver-memory $driver_memory \
   --executor-memory $executor_memory \
   --master local[*] \
   --deploy-mode client \
   --conf spark.network.timeout=800 \
   --conf spark.driver.memoryOverhead=$driver_memory_overhead \
   --conf spark.executor.memoryOverhead=$executor_memory_overhead \
   --files $METAGENOME2VEC_PATH/metagenome2vec/utils/transformation_ADN.cpython-38-x86_64-linux-gnu.so \
   $path_script \
   -mo $mode \
   -np $num_partitions \
   -pd $path_data"/k_"$k"_s_"$s \
   -pg $path_log \
   -o
done
