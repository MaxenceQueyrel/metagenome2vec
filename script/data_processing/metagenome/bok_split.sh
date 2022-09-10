#!/bin/bash

. $METAGENOME2VEC_PATH/metagenome2vec/utils/bash_manager.sh

args $0 "$@"

path_script="$METAGENOME2VEC_PATH/main.py bok_split"

if [[ $help == "true" ]]
then
  python $path_script --help
  exit 0
fi

eval $(parse_yaml $conf_file)


for file in $path_data/*; do
  filename=`basename $file`
  if [ $overwrite = "True" ] || [ ! -d $path_save/$filename ]
  then
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
     -k $k \
     -s $s \
     -np $num_partitions \
     -pd $path_data/$filename \
     -ps $path_save \
     -pg $path_log \
     -o
  fi
done


