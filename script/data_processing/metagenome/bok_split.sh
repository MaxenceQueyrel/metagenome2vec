#!/bin/bash

. $METAGENOME2VEC_PATH/metagenome2vec/utils/bash_manager.sh

args $0 "$@"

if [ -z "$conf_file" ]
then
  usage
  exit
fi

eval $(parse_yaml $conf_file)

for file in $path_data/*; do
  filename=`basename $file`
  if [ $overwrite = "True" ] || [ ! -d $path_save/$filename ]
  then
    spark-submit \
     --conf spark.app.name=bok_split \
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
     $METAGENOME2VEC_PATH/metagenome2vec/data_processing/metagenome/bok_split.py \
     -mo $mode \
     -k $k \
     -s $s \
     -np $num_partitions \
     -pd $path_data/$filename \
     -ps $path_save \
     -pg $path_log \
     -o
  fi
done

