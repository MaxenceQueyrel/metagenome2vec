# -*- coding: latin-1 -*-

import os
import time
import re
import logging

from metagenome2vec.utils import file_manager, spark_manager, parser_creator


# Fast without saving tmp matrix
def bok_merge(spark, path_data, nb_metagenome, num_partitions, mode, overwrite):
    """
    Compute the merge between each files
    :param spark: SparkSession
    :param L_df_file: List, like [folder_name_1, folder_name_2, ...]
    :param mode: String, hdfs s3 or local
    :return:
    """
    # select a number of data file (fastq and fasta) we want to read
    L_df_file = file_manager.generate_list_file(path_data, mode, False)

    saving_mode = "overwrite" if args.overwrite else None
    df_word_count_tmp = None
    if "bok.parquet" in [os.path.basename(x) for x in L_df_file] and not overwrite:
        logging.info("Already computed")
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
        logging.info("path: %s" % path_word_count_tmp)
        logging.info("%s %s\n" % (i, str(os.path.basename(df_file))))
        df_word_count = spark.read.parquet(df_file)
        # Merge the ancient df with the new one
        df_word_count_tmp = spark_manager.merge_df_count(df_word_count_tmp, df_word_count, ["kmer"], num_partitions)
        df_word_count_tmp = df_word_count_tmp.persist()
        df_word_count_tmp.count()
        df_word_count_tmp.write.save(path_word_count_tmp, format="parquet", mode="overwrite")
        file_manager.remove_dir(os.path.join(path_data, "bok_%s.parquet" % (i - 1)), mode)
        logging.info("Iteration %s : %s en %s" % (i, df_file, time.time() - d))
        spark.catalog.clearCache()
    path_word_count = os.path.join(path_data, "bok.parquet")
    df_word_count = spark_manager.df_word_add_index(df_word_count_tmp, num_partitions=num_partitions)
    df_word_count.write.save(path_word_count, mode=saving_mode, format="parquet")
    file_manager.remove_dir(path_word_count_tmp, mode)


if __name__ == "__main__":
    args = parser_creator.ParserCreator().parser_bok_merge()

    logging.basicConfig(filename=os.path.join(args.path_log, args.log_file), level=logging.DEBUG)

    spark = spark_manager.createSparkSession("BoK merge")

    logging.info("Start computing")
    bok_merge(spark, args.path_data, args.nb_metagenome, args.num_partitions, args.mode, args.overwrite)
    logging.info("End computing")
