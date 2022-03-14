import os
import logging
from metagenome2vec.utils import file_manager, spark_manager, parser_creator


def preprocess_metagenomic_data(path_data, path_save, spark, n_sample_load=-1, mode="local",
                                overwrite=False, num_partitions=None, in_memory=True):
    """
    Create a sample of path_data_hdfs. Can create multiple sample if path_data_hdfs is a folder.
    :param path_data: String, the path where are stored the data on hdfs
    :param spark: sqlContext, current spark sql context
    :param n_sample_load: long, Maximum number of reads loaded for one sample
    :param path_save: String, path where are stored the data
    :param mode: String, hdfs, local or s3
    :param num_partitions: int, number of partition for the parquet file
    :return:
    """
    saving_mode = "overwrite" if overwrite else None
    if overwrite is False and file_manager.dir_exists(path_save, mode):
        logging.info("%s already exists" % path_save)
        return
    logging.info("Begin preprocessing of %s" % path_save)
    df = spark_manager.read_raw_fastq_file_to_df(spark, path_data, n_sample_load=n_sample_load,
                                                 num_partitions=num_partitions, in_memory=in_memory)
    df.write.save(path_save, mode=saving_mode, format="parquet")
    logging.info("End preprocessing")


if __name__ == "__main__":
    args = parser_creator.ParserCreator().parser_clean_raw_data()
    logging.basicConfig(filename=os.path.join(args.path_log, args.log_file), level=logging.INFO)
    logging.getLogger('pyspark').setLevel(logging.ERROR)
    logging.getLogger("py4j").setLevel(logging.ERROR)

    spark = spark_manager.createSparkSession("clean_raw_data")

    preprocess_metagenomic_data(args.path_data, args.path_save, spark, args.n_sample_load, args.mode,
                                overwrite=args.overwrite, num_partitions=args.num_partitions, in_memory=args.in_memory)
