import os
import time
import re
import logging
from pyspark import SparkContext
from metagenome2vec.utils import file_manager, spark_manager, transformation_ADN


def preprocess_metagenomic_data(
    spark: SparkContext,
    path_data: str,
    path_save: str,
    n_sample_load: int = -1,
    num_partitions: int = None,
    mode: str = "local",
    in_memory: bool = True,
    overwrite: bool = False,
):
    """From a fastq sample located at path_data, it creates a parquet file with the preprocessed metagenomic data.
    It can create multiple samples if path_data is a folder.

    Args:
        spark (SparkContext): The current Spark context.
        path_data (str): The path where are located the data.
        path_save (str): The path where the preprocessed data will be saved.
        n_sample_load (int, optional): Maximum number of reads loaded for one sample. Defaults to -1.
        num_partitions (int, optional): number of partition for the parquet file. Defaults to None.
        mode (str, optional): The storage mode between hdfs, local or s3. Defaults to "local".
        in_memory (bool, optional): If True it use memory for the preprocessing. Defaults to True.
        overwrite (bool, optional): If True it will overwrite an existing file or folder at path_save. Defaults to False.
    """
    saving_mode = "overwrite" if overwrite else None
    path_save = os.path.join(path_save, os.path.basename(path_data.strip("/")))
    if overwrite is False and file_manager.dir_exists(path_save, mode):
        logging.info("%s already exists" % path_save)
        return
    logging.info("Begin preprocessing of %s" % path_data)
    df, L_list_tmp_file = spark_manager.read_raw_fastq_file_to_df(
        spark,
        path_data,
        n_sample_load=n_sample_load,
        num_partitions=num_partitions,
        in_memory=in_memory,
    )
    df.write.save(path_save, mode=saving_mode, format="parquet")
    if L_list_tmp_file:
        for tmp_file_name in L_list_tmp_file:
            os.remove(tmp_file_name)
    logging.info("End preprocessing")


def bok_split(
    spark: SparkContext,
    path_data: str,
    path_save: str,
    k_mer_size: int,
    step: int,
    mode: str = "local",
    num_partitions: int = 50,
    overwrite: bool = False,
):
    """Split each read into a coutning of kmers.

    Args:
        spark (SparkContext): _description_
        path_data (str): _description_
        path_save (str): _description_
        k_mer_size (int): _description_
        step (int): _description_
        mode (str, optional): _description_. Defaults to "local".
        num_partitions (int, optional): _description_. Defaults to 50.
        overwrite (bool, optional): _description_. Defaults to False.
    """

    saving_mode = "overwrite" if overwrite else None
    path_save = os.path.join(path_save, "k_%s_s_%s" % (k_mer_size, step))
    file_manager.create_dir(path_save, mode)
    # We compute the first folder containing the first metagenome which will be merged with the
    # result of the other folder to create a big database
    # All computation are saved for each folder
    logging.info("Computing BoK for file %s\n" % path_data)
    path_bok = os.path.join(path_save, os.path.basename(path_data.strip("/")))
    # Check if files already exists, if overwrite is False return the already computed files
    if overwrite is False:
        if file_manager.dir_exists(path_bok, mode):
            return
    # Reading, cleaning and concatanation for each file in list_rdd_file
    df = spark.read.parquet(path_data)
    # Create the df context
    df_word_count = spark_manager.df_to_df_word_count(
        df, k_mer_size, step, num_partitions=num_partitions
    )
    logging.info("Saving df bok")
    df_word_count.write.save(path_bok, mode=saving_mode, format="parquet")
    logging.info("df bok saved")


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

    saving_mode = "overwrite" if overwrite else None
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
        df_word_count_tmp = spark_manager.merge_df_count(
            df_word_count_tmp, df_word_count, ["kmer"], num_partitions
        )
        df_word_count_tmp = df_word_count_tmp.persist()
        df_word_count_tmp.count()
        df_word_count_tmp.write.save(
            path_word_count_tmp, format="parquet", mode="overwrite"
        )
        file_manager.remove_dir(
            os.path.join(path_data, "bok_%s.parquet" % (i - 1)), mode
        )
        logging.info("Iteration %s : %s en %s" % (i, df_file, time.time() - d))
        spark.catalog.clearCache()
    path_word_count = os.path.join(path_data, "bok.parquet")
    df_word_count = spark_manager.df_word_add_index(
        df_word_count_tmp, num_partitions=num_partitions
    )
    df_word_count.write.save(path_word_count, mode=saving_mode, format="parquet")
    file_manager.remove_dir(path_word_count_tmp, mode)


def kmerize_metagenomic_data(
    spark: SparkContext,
    path_data: str,
    path_save: str,
    k_mer_size: int,
    n_sample_load=-1,
    num_partitions: int = None,
    in_memory: bool = True,
):
    """

    Args:
        spark (SparkContext): current spark sql context
        path_data (str): the path where are stored the data
        path_save (str): path where are stored the data
        k_mer_size (int): _description_
        n_sample_load (int, optional): Maximum number of reads loaded for one sample. Defaults to -1.
        num_partitions (int, optional): number of partition for the parquet file. Defaults to None.
        in_memory (bool, optional): _description_. Defaults to True.
    """
    col_name = "read"
    open_mode = "a" if os.path.exists(path_save) else "w"
    with open(path_save, open_mode) as f_res:
        df = spark_manager.read_raw_fastq_file_to_df(
            spark,
            path_data,
            n_sample_load=n_sample_load,
            num_partitions=num_partitions,
            in_memory=in_memory,
        )
        L_reads = [x[0] for x in df.select(col_name).collect()]
        transformation_ADN.cut_and_write_reads(
            L_reads, f_res, k_mer_size, s=1, mode="c", remove_unk=False
        )
