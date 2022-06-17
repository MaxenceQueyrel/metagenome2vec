#!/bin/bash

. $METAGENOME2VEC_PATH/metagenome2vec/utils/bash_manager.sh

args $0 "$@"

path_script="$METAGENOME2VEC_PATH/metagenome2vec/data_processing/simulation/create_simulated_metagenome2vec_dataset.py"

if [[ $help == "true" ]]
then
  python $path_script --help
  exit 0
fi

eval $(parse_yaml $conf_file)

if [[ $overwrite == "true" ]]
then
  path_script=$path_script" -o"
fi

python $path_script -pd $path_data -ps $path_save

for file in $path_save/*; do
  filename=`basename $file`
  if [[ $filename != metadata.csv ]]
  then
    spark-submit \
    --conf spark.app.name=clean_raw_data \
    --conf spark.locality.wait=0 \
    --conf spark.sql.autoBroadcastJoinThreshold=-1 \
    --conf spark.scheduler.minRegisteredResourcesRatio=1 \
    --conf spark.executor.extraLibraryPath="$METAGENOME2VEC_PATH/metagenome2vec/utils/transformation_ADN.cpython-38-x86_64-linux-gnu.so" \
    --conf spark.cleaner.referenceTracking=false \
    --conf spark.cleaner.referenceTracking.blocking=false \
    --conf spark.cleaner.referenceTracking.blocking.shuffle=false \
    --conf spark.cleaner.referenceTracking.cleanCheckpoints=false \
    --num-executors $num_executors \
    --executor-cores $executor_cores \
    --driver-memory $driver_memory \
    --executor-memory $executor_memory \
    --master local[*] \
    --conf spark.network.timeout=800 \
    --conf spark.driver.memoryOverhead=$driver_memory_overhead \
    --conf spark.executor.memoryOverhead=$executor_memory_overhead \
    --files $METAGENOME2VEC_PATH/metagenome2vec/utils/transformation_ADN.cpython-38-x86_64-linux-gnu.so \
    $METAGENOME2VEC_PATH/metagenome2vec/data_processing/metagenome/clean_raw_data.py \
    -nsl $n_sample_load \
    -pd $path_save/$filename \
    -ps $path_save/${filename}.parquet \
    -mo $mode \
    -np $num_partitions \
    -pg $METAGENOME2VEC_PATH/logs/clean_raw_data/ \
    -o \
    -im

    mv $path_save/${filename}.parquet $path_save/$filename
  fi
done

