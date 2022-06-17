#!/bin/bash

. $METAGENOME2VEC_PATH/metagenome2vec/utils/bash_manager.sh

args $0 "$@"

path_script="$METAGENOME2VEC_PATH/metagenome2vec/read2genome/train_h2o_model.py"

if [[ $help == "true" ]]
then
  python $path_script --help
  exit 0
fi

eval $(parse_yaml $conf_file)

path_data="${input},${labels}"

spark-submit \
 --class water.SparklingWaterDriver \
 --conf "spark.executor.extraClassPath=-Dhdp.version=current" \
 --conf spark.app.name=read2genome \
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
 --conf spark.network.timeout=800 \
 --conf spark.driver.memoryOverhead=$driver_memory_overhead \
 --conf spark.executor.memoryOverhead=$executor_memory_overhead \
 --files $METAGENOME2VEC_PATH/metagenome2vec/utils/transformation_ADN.cpython-38-x86_64-linux-gnu.so \
 $path_script \
 -pd $path_data \
 -ps /user/mqueyrel/data/simulated/ \
 -rv $read2vec \
 -pg $path_tmp_folder \
 -mo $mode \
 -np $num_partitions \
 -k $k \
 -tl $tax_level \
 -pm $path_model \
 -f $f_name \
 -mla $machine_learning_algorithm \
 -mm $max_model \
 -nf $n_fold \
 -nsl $n_sample_load \
 -prv $path_read2vec \
 -pmwc $path_metagenome_word_count \
 -tt $tax_taken
