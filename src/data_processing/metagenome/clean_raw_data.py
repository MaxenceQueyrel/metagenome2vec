from pyspark.sql import types as T
import os
import sys
import subprocess
root_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.insert(0, os.path.join(root_folder, "utils"))
import logger
import hdfs_functions as hdfs
import parser_creator
from string_names import *
SEED = 42


def preprocess_metagenomic_data(path_data, path_save, spark, n_sample_load=-1, mode="hdfs",
                                is_file=False, saving_mode="overwrite",
                                num_partitions=None):
    """
    Create a sample of path_data_hdfs. Can create multiple sample if path_data_hdfs is a folder.
    :param path_data: String, the path where are stored the data on hdfs
    :param spark: sqlContext, current spark sql context
    :param n_sample_load: long, Maximum number of reads loaded for one sample
    :param path_save: String, path where are stored the data
    :param mode: String, hdfs, local or s3
    :param is_file: boolean (default=False) if True then create a sample of the unique file with the suffix like
    :param saving_mode: str, saving mode, default='overwrite'
    :param num_partitions: int, number of partition for the parquet file
    :return:
    """
    if is_file is False:
        hdfs.create_dir(os.path.dirname(path_save), mode, os.path.basename(path_save))
        list_data = hdfs.generate_list_file(path_data, mode)
        if file_to_del is not None:
            list_data = [x for x in list_data if x[0] not in file_to_del]
        for i, folder_files in enumerate(list_data):
            folder = folder_files[0]
            log.write("%s: %s" % (i, folder))
            if overwrite is False and hdfs.dir_exists(os.path.join(path_save, folder), mode):
                log.write("%s already exists" % folder)
                continue
            log.writeExecutionTime()
            df = hdfs.read_raw_fastq_file_to_df(spark, os.path.join(path_data, folder), n_sample_load=n_sample_load,
                                                num_partitions=num_partitions, in_memory=in_memory)
            df.write.save(os.path.join(path_save, folder), mode=saving_mode, format="parquet")
            log.writeExecutionTime()
    else:
        if overwrite is False and hdfs.dir_exists(path_save, mode):
            log.write("%s already exists" % path_save)
            return
        df = hdfs.read_raw_fastq_file_to_df(spark, path_data, n_sample_load=n_sample_load,
                                            num_partitions=num_partitions, in_memory=in_memory)
        df.write.save(path_save, mode=saving_mode, format="parquet")



if __name__ == "__main__":

    ###################################
    # ------ Script's Parameters ------#
    ###################################

    parser = parser_creator.ParserCreator()
    args = parser.parser_clean_raw_data()
    n_sample_load = args.n_sample_load
    path_data = args.path_data
    path_save = args.path_save
    is_file = args.is_file
    mode = args.mode
    log_file = args.log_file
    path_log = args.path_log
    overwrite = args.overwrite
    in_memory = args.in_memory
    saving_mode = "overwrite" if args.overwrite else None
    num_partitions = args.num_partitions
    file_to_del = None if args.file_to_del == "None" or args.file_to_del is None else args.file_to_del.split(",")
    log = logger.Logger(path_log, log_file, log_file, **vars(args))

    ###################################
    # ---------- Spark Conf -----------#
    ###################################

    spark = hdfs.createSparkSession("metagenomic_preprocessing")

    ###################################
    # -------------- Run -------------#
    ###################################

    preprocess_metagenomic_data(path_data, path_save, spark, n_sample_load, mode, is_file,
                                saving_mode=saving_mode, num_partitions=num_partitions)
