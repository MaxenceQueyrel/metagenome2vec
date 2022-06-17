#!/bin/bash

usage()
{
    echo "mandatory parameters : --path-data, --path-save, --path-model, --read2genome, --path-metadata"
}

num_executors=2
executor_cores=45
driver_memory=70g
executor_memory=60g
driver_memory_overhead=130g
executor_memory_overhead=60g
path_tmp_folder=$TMP
n_cpus=16
tax_level=species
n_sample_load=-1
mode=hdfs
path_logs=$DEEPGENE/logs/analyse_read2genome

function args()
{
    options=$(getopt \
    --long num-executors: \
    --long executor-cores: \
    --long driver-memory: \
    --long executor-memory: \
    --long driver-memory-overhead: \
    --long executor-memory-overhead: \
    --long read2genome: \
    --long path-data: \
    --long path-model: \
    --long path-save: \
    --long path-tmp-folder: \
    --long tax-level: \
    --long n-sample-load: \
    --long n-cpus: \
    --long path-logs \
    --long path-meta-data \
    -- "$@")
    [ $? -eq 0 ] || {
        echo "Incorrect option provided"
        exit 1
    }
    eval set -- "$options"
    while true; do
        case "$1" in
        --num-executors)
            shift;
            num_executors=$1
            ;;
        --executor-cores)
            shift;
            executor_cores=$1
            ;;
        --driver-memory)
            shift;
            driver_memory=$1
            ;;
        --executor-memory)
            shift;
            executor_memory=$1
            ;;
        --driver-memory-overhead)
            shift;
            driver_memory_overhead=$1
            ;;
        --executor-memory-overhead)
            shift;
            executor_memory_overhead=$1
            ;;
        --read2genome)
            shift;
            read2genome=$1
            ;;
        --path-data)
            shift;
            path_data=$1
            ;;
        --path-model)
            shift;
            path_model=$1
            ;;
        --path-save)
            shift;
            path_save=$1
            ;;
        --path-tmp-folder)
            shift;
            path_tmp_folder=$1
            ;;
        --tax-level)
            shift;
            tax_level=$1
            ;;
        --n-sample-load)
            shift;
            n_sample_load=$1
            ;;
        --n-cpus)
            shift;
            n-cpus=$1
            ;;
        --path-logs)
            shift;
            path_logs=$1
            ;;
        --path-metadata)
            shift;
            path_metadata=$1
            ;;
        --)
            shift
            break
            ;;
        esac
        shift
    done
}

args $0 "$@"

if [ -z "$path_model" ] || [ -z "$path_save" ] || [ -z "$path_data" ] || [ -z "$read2genome" ] || [ -z "$path_metadata"]
then
  usage
  exit
fi

spark-submit \
 --class water.SparklingWaterDriver \
 --conf "spark.executor.extraClassPath=-Dhdp.version=current" \
 --jars /share/apps/sparkling-water-3.28.0.1-1-2.3/assembly/build/libs/sparkling-water-assembly_2.11-3.28.0.1-1-2.3-all.jar \
 --conf spark.app.name=read2genome \
 --conf spark.locality.wait=0 \
 --conf spark.sql.autoBroadcastJoinThreshold=-1 \
 --conf spark.scheduler.minRegisteredResourcesRatio=1 \
 --conf spark.executor.extraLibraryPath="$DEEPGENE/Pipeline/utils/transformation_ADN.so" \
 --conf spark.cleaner.referenceTracking=false \
 --conf spark.cleaner.referenceTracking.blocking=false \
 --conf spark.cleaner.referenceTracking.blocking.shuffle=false \
 --conf spark.cleaner.referenceTracking.cleanCheckpoints=false \
 --conf spark.rpc.message.maxSize=1024 \
 --num-executors $num_executors \
 --executor-cores $executor_cores \
 --driver-memory $driver_memory \
 --executor-memory $executor_memory \
 --master yarn \
 --conf spark.network.timeout=800 \
 --conf spark.driver.memoryOverhead=$driver_memory_overhead \
 --conf spark.executor.memoryOverhead=$executor_memory_overhead \
 --conf spark.sql.execution.arrow.maxRecordsPerBatch=5000 \
 --py-files $DEEPGENE/Pipeline/utils/hdfs_functions.py,\
$DEEPGENE/Pipeline/utils/parser_creator.py,\
$DEEPGENE/Pipeline/read2vec/read2vec.py,\
$DEEPGENE/Pipeline/read2vec/basic.py,\
$DEEPGENE/Pipeline/read2vec/seq2seq.py,\
$DEEPGENE/Pipeline/read2vec/SIF.py,\
$DEEPGENE/Pipeline/utils/transformation_ADN2.py,\
$DEEPGENE/Pipeline/utils/data_manager.py\
 --files $DEEPGENE/Pipeline/utils/transformation_ADN.so \
 $DEEPGENE/Pipeline/analyse/analyse_read2genome.py \
 -pd $path_data \
 -pg $DEEPGENE/logs/analyse_read2genome \
 -mo $mode \
 -tl $tax_level \
 -pm $path_model \
 -nsl $n_sample_load \
 -nc $n_cpus \
 -rg $read2genome \
 -pt $path_tmp_folder \
 -pmd $path_metadata

