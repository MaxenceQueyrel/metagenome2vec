#!/bin/bash

usage()
{
    echo "mandatory parameters : -k, --path-data, --path-read2vec, --path_save"
}

num_executors=10
executor_cores=5
driver_memory=20g
executor_memory=20g
driver_memory_overhead=20g
executor_memory_overhead=10g
num_partitions=200
tax_level=species
path_metagenome_word_count=None
read2vec=basic
path_genome_distance=$DEEPGENE/data/taxonomy/matrix_dist.csv
max_length=100
path_tmp_folder=$TMP
n_cpus=45
id_gpu=-1
path_metadata=$DEEPGENE/data/taxonomy/tax_report.csv

function args()
{
    options=$(getopt -o k:f: \
    --long num-executors: \
    --long executor-cores: \
    --long driver-memory: \
    --long executor-memory: \
    --long driver-memory-overhead: \
    --long executor-memory-overhead: \
    --long num-partitions: \
    --long tax-level: \
    --long read2vec: \
    --long path-data: \
    --long path-save: \
    --long path-read2vec: \
    --long read2vec: \
    --long path-metagenome-word-count: \
    --long n-cpus \
    --long path-tmp-folder \
    --long id-gpu \
    --long path-metadata \
    --long max-length \
    --long path-genome-distance \
    -- "$@")
    [ $? -eq 0 ] || {
        echo "Incorrect option provided"
        exit 1
    }
    eval set -- "$options"
    while true; do
        case "$1" in
        -k)
            shift;
            k=$1
            ;;
        -f)
            shift;
            f=$1
            ;;
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
        --num-partitions)
            shift;
            num_partitions=$1
            ;;
        --tax-level)
            shift;
            tax_level=$1
            ;;
        --read2vec)
            shift;
            read2vec=$1
            ;;
        --path-data)
            shift;
            path_data=$1
            ;;
        --path-save)
            shift;
            path_save=$1
            ;;
        --path-read2vec)
            shift;
            path_read2vec=$1
            ;;
        --path-metagenome-word-count)
            shift;
            path_metagenome_word_count=$1
            ;;
        --machine-learning-algorithm)
            shift;
            machine_learning_algorithm=$1
            ;;
        --n-cpus)
            shift;
            n_cpus=$1
            ;;
        --path-tmp-folder)
            shift;
            path_tmp_folder=$1
            ;;
        --id-gpu)
            shift;
            id_gpu=$1
            ;;
        --path-metadata)
            shift;
            path_metadata=$1
            ;;
        --max-length)
            shift;
            max_length=$1
            ;;
        --path-genome-distance)
            shift;
            path_genome_distance=$1
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

if [ -z "$k" ] || [ -z "$path_data" ] || [ -z "$path_read2vec" ] || [ -z "$path_save" ]
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
 --conf spark.cleaner.referenceTracking=True \
 --conf spark.cleaner.referenceTracking.blocking=True \
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
 --py-files $DEEPGENE/Pipeline/utils/hdfs_functions.py,\
$DEEPGENE/Pipeline/utils/parser_creator.py,\
$DEEPGENE/Pipeline/read2vec/read2vec.py,\
$DEEPGENE/Pipeline/read2vec/basic.py,\
$DEEPGENE/Pipeline/read2vec/transformer.py,\
$DEEPGENE/Pipeline/read2vec/SIF.py,\
$DEEPGENE/Pipeline/utils/transformation_ADN2.py,\
$DEEPGENE/Pipeline/utils/data_manager.py\
 --files $DEEPGENE/Pipeline/utils/transformation_ADN.so \
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
 -pgd $path_genome_distance \
 -np $num_partitions

