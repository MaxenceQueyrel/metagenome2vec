from pyspark.sql import types as T
import os
import sys
root_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.insert(0, os.path.join(root_folder, "utils"))
import logger
import hdfs_functions as hdfs
import parser_creator
if sys.version_info[0] == 3 and sys.version_info[1] == 7:
    import transformation_ADN
else:
    import transformation_ADN2 as transformation_ADN

SEED = 42


def kmerized_metagenomic_data(path_data, path_save, spark, n_sample_load=-1, mode="hdfs",
                                is_file=False, num_partitions=None):
    """
    Create a sample of path_data_hdfs. Can create multiple sample if path_data_hdfs is a folder.
    :param path_data: String, the path where are stored the data on hdfs
    :param spark: sqlContext, current spark sql context
    :param n_sample_load: long, Maximum number of reads loaded for one sample
    :param path_save: String, path where are stored the data
    :param mode: String, hdfs, local or s3
    :param is_file: boolean (default=False) if True then create a sample of the unique file with the suffix like
    :param num_partitions: int, number of partition for the parquet file
    :return:
    """
    col_name = "read"
    with open(path_save, "w") as f_res:
        if is_file is False:
            list_data = hdfs.generate_list_file(path_data, mode)
            for i, folder_files in enumerate(list_data):
                folder = folder_files[0]
                log.write("%s : %s" % (i, folder))
                log.writeExecutionTime()
                log.writeExecutionTime()
                df = hdfs.read_raw_fastq_file_to_df(spark, os.path.join(path_data, folder), n_sample_load=n_sample_load,
                                                    num_partitions=num_partitions, in_memory=in_memory)
                L_reads = [x[0] for x in df.select(col_name).collect()]
                transformation_ADN.cut_and_write_reads(L_reads, f_res, k_mer_size, s=1, mode="c", remove_unk=False)
                log.writeExecutionTime()
        else:
            df = hdfs.read_raw_fastq_file_to_df(spark, path_data, n_sample_load=n_sample_load,
                                                num_partitions=num_partitions, in_memory=in_memory)
            L_reads = [x[0] for x in df.select(col_name).collect()]
            transformation_ADN.cut_and_write_reads(L_reads, f_res, k_mer_size, s=1, mode="c", remove_unk=False)


if __name__ == "__main__":

    ###################################
    # ------ Script's Parameters ------#
    ###################################

    parser = parser_creator.ParserCreator()
    args = parser.parser_metagenomic_kmerization()
    n_sample_load = args.n_sample_load
    path_data = args.path_data
    path_save = args.path_save
    k_mer_size = args.k_mer_size
    is_file = args.is_file
    mode = args.mode
    log_file = args.log_file
    path_log = args.path_log
    num_partitions = args.num_partitions
    in_memory = args.in_memory
    log = logger.Logger(path_log, log_file, log_file, **vars(args))

    ###################################
    # ---------- Spark Conf -----------#
    ###################################

    spark = hdfs.createSparkSession("metagenomic_kmerization")

    ###################################
    # -------------- Run -------------#
    ###################################

    kmerized_metagenomic_data(path_data, path_save, spark, n_sample_load, mode, is_file, num_partitions=num_partitions)
