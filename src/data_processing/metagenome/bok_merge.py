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


# Fast without saving tmp matrix
def compute(spark, L_df_file):
    """
    Compute the merge between each files
    :param spark: SparkSession
    :param L_df_file: List, like [folder_name_1, folder_name_2, ...]
    :param mode: String, hdfs s3 or local
    :return:
    """
    df_word_count_tmp = None
    if "bok.parquet" in [os.path.basename(x) for x in L_df_file] and not overwrite:
        log.write("Already computed")
        return
    index_tmp_file = -1
    if not overwrite:
        for df_file in [os.path.basename(x) for x in L_df_file]:
            if re.match("^bok", df_file):
                index_tmp_file = int(re.sub("^bok_([0-9]*)\..*$", "\\1", df_file))
    if index_tmp_file > -1:
        path_word_count_tmp = os.path.join(path_data, "bok_%s.parquet" % index_tmp_file)
        df_word_count_tmp = spark.read.parquet(path_word_count_tmp)
    for i, df_file in enumerate(L_df_file):
        if i <= index_tmp_file:
            continue
        if i > nb_metagenome:
            break
        d = time.time()
        path_word_count_tmp = os.path.join(path_data, "bok_%s.parquet" % i)
        log.write(path_word_count_tmp)
        log.write("%s %s\n" % (i, str(os.path.basename(df_file))))
        df_word_count = spark.read.parquet(df_file)
        # Merge the ancient df with the new one
        df_word_count_tmp = hdfs.merge_df_count(df_word_count_tmp, df_word_count, ["kmer"], num_partitions)
        df_word_count_tmp = df_word_count_tmp.persist()
        df_word_count_tmp.count()
        df_word_count_tmp.write.save(path_word_count_tmp, format="parquet", mode="overwrite")
        hdfs.remove_dir(os.path.join(path_data, "bok_%s.parquet" % (i - 1)), mode)
        log.write("Iteration %s : %s en %s" % (i, df_file, time.time() - d))
        for (id, rdd) in spark.sparkContext._jsc.getPersistentRDDs().items():
            rdd.unpersist()
    path_word_count = os.path.join(path_data, "bok.parquet")
    df_word_count = hdfs.df_word_add_index(df_word_count_tmp, num_partitions=num_partitions)
    df_word_count.write.save(path_word_count, mode=saving_mode, format="parquet")
    hdfs.remove_dir(path_word_count_tmp, mode)


if __name__ == "__main__":

    ###################################
    # ------ Script's Parameters ------#
    ###################################
    parser = parser_creator.ParserCreator()
    args = parser.parser_bok_merge()
    # We define the name of the folder where we store the results of the heat map
    num_partitions = args.num_partitions
    mode = args.mode
    s3_bucket = re.sub("\/*$", "", args.bucket)
    log_file = args.log_file
    path_log = args.path_log
    overwrite = args.overwrite
    saving_mode = "overwrite" if args.overwrite else None
    nb_metagenome = args.nb_metagenome
    log = logger.Logger(path_log, log_file, log_file,
                        variable_time={"mode": "merge"},
                        **vars(args))
    name_df_word_count = "df_word_count.parquet"
    path_data = args.path_data

    ###################################
    # ---------- Spark Conf -----------#
    ###################################

    spark = hdfs.createSparkSession("Structuration learning")

    ###################################
    # ---------- List Files -----------#
    ###################################

    # select a number of data file (fastq and fasta) we want to read
    L_df_file = hdfs.generate_list_file(path_data, mode, False)

    ###################################
    # ---- Compute transformation -----#
    # ---- And Saving Data ------------#
    ###################################

    log.writeExecutionTime()
    compute(spark, L_df_file)
    log.writeExecutionTime("Main run")
    log.close()
