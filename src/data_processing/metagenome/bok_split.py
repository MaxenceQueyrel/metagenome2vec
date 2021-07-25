# -*- coding: latin-1 -*-

import os
import time
import re
import sys
root_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.insert(0, os.path.join(root_folder, "utils"))
import logger
import parser_creator
import hdfs_functions as hdfs


def compute(df_name, mode):
    """
    Compute the structuring for one metagenome
    :param df_name: str, Name of the dataframe
    :param mode: String, hdfs, s3 or local
    :return: rdd_context, rdd_word_count
    """
    path_bok = os.path.join(path_save, os.path.basename(df_name))
    # Check if files already exists, if overwrite is False return the already computed files
    if overwrite is False:
        if hdfs.dir_exists(path_bok, mode):
            return
    # Reading, cleaning and concatanation for each file in list_rdd_file
    df = spark.read.parquet(df_name)
    # Create the df context
    df_word_count = hdfs.df_to_df_word_count(df, k, s, num_partitions=num_partitions)
    log.write("Saving df bok")
    df_word_count.write.save(path_bok, mode=saving_mode, format="parquet")
    log.write("df bok saved")


def compute_bok_split(spark, L_df_name, mode="hdfs"):
    """
    Compute the transformation for each metagenome and save it
    :param spark: SparkSession
    :param L_df_name: List, contains list of all files name that we want to compute
            format : ((folder_name_1, (file_1, file_2...), (folder_name_2, (file_1, file_2,..)))
    :param mode: String, hdfs, s3 or local
    """
    # We compute the first folder containing the first metagenome which will be merged with the
    # result of the other folder to create a big database
    # All computation are saved for each folder
    for i, df_name in enumerate(L_df_name):
        if i == nb_metagenome:
            break
        d = time.time()
        log.write("%s %s\n" % (i, df_name))
        compute(df_name, mode)
        log.write("Iteration %s: %s in %s" % (i, df_name, time.time()-d))
        spark.catalog.clearCache()


if __name__ == "__main__":

    ####################################
    # ------ Script's Parameters ------#
    ####################################

    parser = parser_creator.ParserCreator()
    args = parser.parser_bok_split()
    k = args.k_mer_size
    s = args.step
    is_file = args.is_file
    # We define the name of the folder where we store the results of the heat map
    num_partitions = args.num_partitions
    mode = args.mode
    s3_bucket = re.sub("\/*$", "", args.bucket)
    path_data = args.path_data
    log_file = args.log_file
    overwrite = args.overwrite
    saving_mode = "overwrite" if overwrite else None
    str_param = "k_%s_s_%s" % (k, s)
    path_save = os.path.join(args.path_save, str_param)
    path_log = args.path_log
    nb_metagenome = args.nb_metagenome
    log = logger.Logger(path_log, log_file, log_file,
                        variable_time={"k": k, "s": s, "mode": "split"},
                        **vars(args))
    name_df_word_count = "df_word_count.parquet"

    ####################################
    # ---------- Spark Conf -----------#
    ####################################

    spark = hdfs.createSparkSession("data processing bok split")

    ###################################
    # ---------- List Files ----------#
    ###################################

    # select a number of data file (fastq and fasta) we want to read
    if not is_file:
        L_df_name = hdfs.generate_list_file(path_data, mode, False)
    else:
        L_df_name = [path_data]

    ###################################
    # ---- Compute transformation -----#
    # ---- And Saving Data ------------#
    ###################################

    log.writeExecutionTime()
    hdfs.create_dir(path_save, mode)
    compute_bok_split(spark, L_df_name, mode)
    log.writeExecutionTime("Main run")
    log.close()
