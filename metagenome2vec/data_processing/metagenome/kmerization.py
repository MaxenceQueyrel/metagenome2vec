import os
import logging

from metagenome2vec.utils import spark_manager
from metagenome2vec.utils import parser_creator
from metagenome2vec.utils import transformation_ADN


def kmerize_metagenomic_data(spark, path_data, path_save, k_mer_size, n_sample_load=-1,
                             num_partitions=None, in_memory=True):
    """
    Create a sample of path_data_hdfs. Can create multiple sample if path_data_hdfs is a folder.
    :param spark: sqlContext, current spark sql context
    :param path_data: String, the path where are stored the data on hdfs
    :param n_sample_load: long, Maximum number of reads loaded for one sample
    :param path_save: String, path where are stored the data
    :param num_partitions: int, number of partition for the parquet file
    :return:
    """
    col_name = "read"
    open_mode = "a" if os.path.exists(path_save) else "w"
    with open(path_save, open_mode) as f_res:
        df = spark_manager.read_raw_fastq_file_to_df(spark, path_data, n_sample_load=n_sample_load,
                                                     num_partitions=num_partitions, in_memory=in_memory)
        L_reads = [x[0] for x in df.select(col_name).collect()]
        transformation_ADN.cut_and_write_reads(L_reads, f_res, k_mer_size, s=1, mode="c", remove_unk=False)


if __name__ == "__main__":
    args = parser_creator.ParserCreator().parser_metagenomic_kmerization()

    logging.basicConfig(filename=os.path.join(args.path_log, args.log_file), level=logging.DEBUG)

    spark = spark_manager.createSparkSession("Metagenome kmerization")

    kmerize_metagenomic_data(spark, args.path_data, args.path_save, args.k_mer_size, args.n_sample_load,
                             num_partitions=args.num_partitions, in_memory=args.in_memory)
