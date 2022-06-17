#!/bin/bash

. $METAGENOME2VEC_PATH/metagenome2vec/utils/bash_manager.sh

args $0 "$@"

path_script="$METAGENOME2VEC_PATH/metagenome2vec/metagenome_vectorization/embeddings.py"

if [[ $help == "true" ]]
then
  python $path_script --help
  exit 0
fi

eval $(parse_yaml $conf_file)


cpt=0
while read p; do
  IFS=',' read -ra ADDR <<< "$p"
  if [ $cpt -eq 0 ]
  then
    idx=0
    for i in "${ADDR[@]}"; do
      if [ "$i" == "id.fasta" ]
      then
        idx_fasta=$idx
      fi
      if [ "$i" == "group" ]
      then
        idx_group=$idx
      fi
      if [ "$i" == "id.subject" ]
      then
        idx_subject=$idx
      fi
      idx=$((idx+1))
    done
    cpt=$((cpt+1))
  elif [ $cpt -eq $nb_metagenome ]
  then
    break
  else
    id_fasta=${ADDR[$idx_fasta]}
    group=${ADDR[$idx_group]}
    id_subject=${ADDR[$idx_subject]}
    if [ -f $path_save/tmp/$id_fasta ]; then
      continue
    fi
    spark-submit \
      --conf spark.app.name=embeddings \
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
      -ps $path_save \
      -k $k \
      -np $num_partitions \
      -mo local \
      -pd $path_data \
      -T $thresholds \
      -prv $path_read2vec \
      -pt $path_tmp_folder \
      -pg $path_log \
      -rv $read2vec \
      -ig $id_gpu \
      -pmwc $path_metagenome_word_count \
      -ct $computation_type \
      -pmd $path_metadata \
      -prg $path_read2genome \
      -rg $read2genome \
      -pmca $path_metagenome_cut_analyse \
      -pfsr $path_folder_save_read2genome \
      -il $id_fasta,$group \
      -nsl $n_sample_load
    cpt=$((cpt+1))
  fi
done < $path_metadata


if [ $? -ne 0 ]; then
    exit 1
fi


echo "Merge files"
spark-submit \
  --conf spark.app.name=embeddings_merging \
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
  -ps $path_save \
  -k $k \
  -np $num_partitions \
  -mo local \
  -pd $path_data \
  -T $thresholds \
  -prv $path_read2vec \
  -pt $path_tmp_folder \
  -pg $path_log \
  -rv $read2vec \
  -ig $id_gpu \
  -pmwc $path_metagenome_word_count \
  -ct 3 \
  -pmd $path_metadata \
  -prg $path_read2genome \
  -rg $read2genome \
  -pmca $path_metagenome_cut_analyse \
  -pfsr $path_folder_save_read2genome \
  -nsl $n_sample_load
