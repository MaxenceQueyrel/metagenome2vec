# -*- coding: latin-1 -*-

import os
import logging

from metagenome2vec.utils import file_manager
from metagenome2vec.utils import spark_manager
from metagenome2vec.utils import parser_creator


def bok_split(spark, path_data, k_mer_size, step, mode="local", num_partitions=50, nb_metagenome=-1, overwrite=False):
    """
    Compute the transformation for each metagenome and save it
    :param spark: SparkSession
    :param L_df_name: List, contains list of all files name that we want to compute
            format : ((folder_name_1, (file_1, file_2...), (folder_name_2, (file_1, file_2,..)))
    :param mode: String, hdfs, s3 or local
    """
    saving_mode = "overwrite" if overwrite else None
    path_save = os.path.join(args.path_save, "k_%s_s_%s" % (k_mer_size, step))
    file_manager.create_dir(path_save, mode)
    # We compute the first folder containing the first metagenome which will be merged with the
    # result of the other folder to create a big database
    # All computation are saved for each folder
    logging.info("Computing file %s\n" % path_data)
    path_bok = os.path.join(path_save, os.path.basename(path_data))
    # Check if files already exists, if overwrite is False return the already computed files
    if overwrite is False:
        if file_manager.dir_exists(path_bok, mode):
            return
    # Reading, cleaning and concatanation for each file in list_rdd_file
    df = spark.read.parquet(path_data)
    # Create the df context
    df_word_count = spark_manager.df_to_df_word_count(df, k_mer_size, step, num_partitions=num_partitions)
    logging.info("Saving df bok")
    df_word_count.write.save(path_bok, mode=saving_mode, format="parquet")
    logging.info("df bok saved")


if __name__ == "__main__":
    args = parser_creator.ParserCreator().parser_bok_split()

    logging.basicConfig(filename=os.path.join(args.path_log, args.log_file), level=logging.DEBUG)

    spark = spark_manager.createSparkSession("BoK split")

    logging.info("Start computing")
    bok_split(spark, args.path_data, args.k_mer_size, args.step, args.mode, args.num_partitions,
              args.nb_metagenome, args.is_file, args.overwrite)
    logging.info("End computing")
