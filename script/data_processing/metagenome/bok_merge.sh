#!/bin/bash

. $METAGENOME2VEC_PATH/metagenome2vec/utils/bash_manager.sh

args $0 "$@"

path_script="$METAGENOME2VEC_PATH/metagenome2vec/data_processing/metagenome/bok_merge.py"

if [[ $help == "True" ]]
then
  python $path_script --help
  exit 0
fi

eval $(parse_yaml $conf_file)

if [[ $overwrite = "True" ]]
then
  rm -r $path_data/k_"$k"_s_"$s"/bok.parquet
fi

while [ ! -d $path_data/k_"$k"_s_"$s"/bok.parquet ]
do
  spark-submit \
   --num-executors $num_executors \
   --executor-cores $executor_cores \
   --driver-memory $driver_memory \
   --executor-memory $executor_memory \
   --master $master \
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
