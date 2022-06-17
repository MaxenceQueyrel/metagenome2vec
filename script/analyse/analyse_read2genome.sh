#PBS -N analyse_read2genome
#PBS -l nodes=1:ppn=40
#PBS -l mem=32GB
#PBS -q icannl
#PBS -o /home/mqueyrel/tmp/analyse_read2genome.out
#PBS -e /home/mqueyrel/tmp/analyse_read2genome.err
#PBS -l walltime=72:00:00

conda activate /home/queyrelm/py3-maxence
pyenv activate py3-maxence

data=$DATA
analyse=$ANALYSE

path_model=${path_model}
path_data_train=${path_data_train}
path_data_valid=${path_data_valid}
path_save=${path_save}
read2genome=${read2genome}
path_metadata=${path_metadata}

driver_memory=${driver_memory}
if [ -z "$driver_memory" ]
then
  driver_memory=32g
fi
driver_memory_overhead=${driver_memory_overhead}
if [ -z "$driver_memory_overhead" ]
then
  driver_memory_overhead=32g
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
  path_tmp_folder=$TMP
fi


spark-submit \
 --conf spark.app.name=read2genome \
 --conf spark.locality.wait=0 \
 --conf spark.sql.autoBroadcastJoinThreshold=-1 \
 --conf spark.scheduler.minRegisteredResourcesRatio=1 \
 --conf spark.executor.extraLibraryPath="$DEEPGENE/Pipeline/utils/transformation_ADN.so" \
 --conf spark.cleaner.referenceTracking=false \
 --conf spark.cleaner.referenceTracking.blocking=false \
 --conf spark.cleaner.referenceTracking.blocking.shuffle=false \
 --conf spark.cleaner.referenceTracking.cleanCheckpoints=false \
 --master local[*] \
 --driver-memory $driver_memory \
 --conf spark.network.timeout=800 \
 --conf spark.driver.memoryOverhead=$driver_memory_overhead \
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