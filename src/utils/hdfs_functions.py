# -*- coding: utf-8 -*-

import subprocess
import os
import sys
if sys.version_info[0] == 3 and sys.version_info[1] == 7:
    import transformation_ADN
else:
    import transformation_ADN2 as transformation_ADN

from pyspark.sql import types as T
from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark import SparkConf
from pyspark.sql import SparkSession

root_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

from string_names import *

################################################
################################################
######## Functions that do not use Spark #######
################################################
################################################


def create_dir(path, mode, sub_path=None):
    """
    Create a dir if not exist

    :param path: String, the dir we want to create if not exists
    :param mode: String : hdfs, local or s3
    :param sub_path: String, slash separated string that compeltes path
    :return:
    """
    if mode == "local" or mode == "s3":
        if sub_path is not None:
            path = os.path.join(path, sub_path)
        if not os.path.exists(path):
            os.makedirs(path)
    if mode == "hdfs":
        if sub_path is None:
            if 0 == dir_exists(path, mode):
                subprocess.call("hdfs dfs -mkdir %s" % path, shell=True)
        else:
            if sub_path[-1] == '/':
                sub_path = sub_path[:-1]
            path_curr = path
            for folder in sub_path.split("/"):
                if 0 == dir_exists(os.path.join(path_curr, folder), mode):
                    subprocess.call("hdfs dfs -mkdir %s" % (os.path.join(path_curr, folder)), shell=True)
                path_curr = os.path.join(path_curr, folder)


def remove_dir(path, mode):
    """
    Remove a directory

    :param path: String, the dir we want to create if not exists
    :param mode: String : hdfs, local or s3
    :return:
    """
    if dir_exists(path, mode):
        if mode == "s3":
            cmd_rm = ["aws", "s3", "rm", "--recursive"]
        elif mode == "local":
            cmd_rm = ["rm", "-r"]
        else:
            cmd_rm = ["hdfs", "dfs", "-rm", "-r"]
        subprocess.call(cmd_rm + [path])


def move_file(path, mode):
    """
    Move a file

    :param path: String, the dir we want to create if not exists
    :param mode: String : hdfs, local or s3
    :return:
    """
    if mode == "s3":
        cmd_mv = ["aws", "s3", "mv", "--recursive"]
    elif mode == "local":
        cmd_mv = ["mv"]
    else:
        cmd_mv = ["hdfs", "dfs", "-mv"]
    subprocess.call(cmd_mv + [path])


def copy_dir(path, mode):
    """
    Copy a directory

    :param path: String, the dir we want to create if not exists
    :param mode: String : hdfs, local or s3
    :return:
    """
    if mode == "s3":
        cmd_cp = ["aws", "s3", "cp", "--recursive"]
    elif mode == "local":
        cmd_cp = ["cp", "-r"]
    else:
        cmd_cp = ["hdfs", "dfs", "-cp"]
    subprocess.call(cmd_cp + [path])


def dir_exists(path, mode):
    """
    Tell if a dir exists or not

    :param path: String
    :param mode: String : hdfs, local or s3
    :return: 1 if exists else 0
    """
    if mode == "hdfs":
        return int(subprocess.check_output('hdfs dfs -test -d %s && echo 1 || echo 0' % path, shell=True))
    if mode == "local":
        return 1 if os.path.isdir(path) else 0
    if mode == "s3":
        try:
            res = subprocess.check_output("aws s3 ls %s" % path, shell=True).split(' ')[-1].replace('\n', '').replace('/', '')
            return int(os.path.basename(path) == res)
        except Exception:
            return 0


def generate_list_file(path_data, mode, sub_folder=True):
    """
    Generate a list of file name which come from the path_data.
    This function follows the architecture of biobank.
    :param path_data: String, the path where are stored the data on hdfs
    :param mode: String : hdfs, local or s3
    :param sub_folder: True, This argument is True when we want to get a list of sub list
    :return:
    l_res : List, list like [sub_folder_name_1, sub_folder_name_2]
    or
    l_res : List, list like [sub_folder_name_1, [path_file_1, path_file_2, path_file_3...],
                                   sub_folder_name_2, ...]
    """
    if mode == "hdfs":
        # create_dir(os.path.join(os.path.dirname(root_folder), "data"), "local")
        # path_subfolder = os.path.join(os.path.dirname(root_folder), "data/list_path_hdfs/%s_subfolder_%s.pkl" % (path_data.split("/")[-1], sub_folder))
        # if os.path.isfile(path_subfolder):
        #    with open(path_subfolder) as f:
        #        return pickle.load(f)
        # Get the list of files / folders in path_data_hdfs
        list_file_hdfs = subprocess.check_output('hdfs dfs -ls %s' % path_data, shell=True)
        list_file_hdfs = list_file_hdfs.decode("utf-8") if isinstance(list_file_hdfs, bytes) else list_file_hdfs
        list_file_hdfs = map(lambda x: x.split(" ")[-1], list_file_hdfs.split("\n")[1:])[:-1]
        # Create a list with the name of the folder we want to del to get only what we want
        if not sub_folder:
            # with open(path_subfolder, 'w') as f:
            #     pickle.dump(list_file_hdfs, f)
            return list_file_hdfs
        l_res = []
        for file_hdfs in list_file_hdfs:
            list_file_hdfs_2 = subprocess.check_output('hdfs dfs -ls %s' % file_hdfs, shell=True)
            list_file_hdfs_2 = map(lambda x: x.split(" ")[-1], list_file_hdfs_2.split("\n")[1:])[:-1]
            list_file_hdfs_2 = list_file_hdfs_2.decode("utf-8") if isinstance(list_file_hdfs_2, bytes) else list_file_hdfs_2
            l_res.append([os.path.basename(file_hdfs), [file_hdfs_2 for file_hdfs_2 in list_file_hdfs_2]])
        # with open(path_subfolder, 'w') as f:
        #     pickle.dump(l_res, f)
        return l_res
    if mode == "s3":
        # Get the list of files / folders in path_data_s3
        list_file_s3 = subprocess.check_output('aws s3 ls %s' % path_data, shell=True)
        list_file_s3 = list_file_s3.decode("utf-8") if isinstance(list_file_s3, bytes) else list_file_s3
        list_file_s3 = [x.split(" ")[-1].replace('/', '') for x in list_file_s3.split('\n')[:-2]]
        # Create a list with the name of the folder we want to del to get only what we want
        if not sub_folder:
            return list_file_s3
        l_res = []
        for file_s3 in list_file_s3:
            list_file_s3_2 = subprocess.check_output('aws s3 ls %s' % os.path.join(path_data, file_s3) + '/', shell=True)
            list_file_s3_2 = list_file_s3_2.decode("utf-8") if isinstance(list_file_s3_2, bytes) else list_file_s3_2
            list_file_s3_2 = [x.split(" ")[-1].replace('/', '') for x in list_file_s3_2.split("\n")[:-1]]
            l_res.append([file_s3, [os.path.join(path_data, file_s3, file_s3_2) for file_s3_2 in list_file_s3_2]])
        return l_res
    if mode == "local":
        # Get the list of files / folders in path_data_hdfs
        list_file = subprocess.check_output('ls %s' % path_data, shell=True)
        list_file = list_file.decode("utf-8") if isinstance(list_file, bytes) else list_file
        list_file = [x.split(" ")[-1] for x in list_file.split("\n")[:-1]]
        # Create a list with the name of the folder we want to del to get only what we want
        if not sub_folder:
            return [os.path.join(path_data, x) for x in list_file]
        l_res = []
        for file in list_file:
            list_file_2 = subprocess.check_output('ls %s' % os.path.join(path_data, file), shell=True)
            list_file_2 = list_file_2.decode("utf-8") if isinstance(list_file_2, bytes) else list_file_2
            list_file_2 = [x.split(" ")[-1] for x in list_file_2.split("\n")[:-1]]
            l_res.append(
                [os.path.basename(file), [os.path.join(path_data, file, file_hdfs_2) for file_hdfs_2 in list_file_2]])
        return l_res


################################################
################################################
########### Functions that use Spark ###########
################################################
################################################

def createSparkSession(name_app, **conf):
    """
    Create a Spark Session
    :param name_app: String
    :param conf: dict the spark context conf
    :return: SparkContext
    """
    conf_ = SparkConf()
    conf_.setAppName(name_app)
    if "master" in conf:
        conf_ = conf_.setMaster(conf["master"])
        del conf["master"]
    if bool(conf):
        conf_ = conf_.setAll(list(conf.items()))
    spark = SparkSession.builder.config(conf=conf_).getOrCreate()
    py_files = [os.path.join(root_folder, "utils", "hdfs_functions.py"),
                os.path.join(root_folder, "utils", "parser_creator.py"),
                os.path.join(root_folder, "utils", "transformation_ADN.cpython-37m-x86_64-linux-gnu.so"),
                os.path.join(root_folder, "utils", "transformation_ADN2.py"),
                os.path.join(root_folder, "utils", "logger.py"),
                os.path.join(root_folder, "utils", "stat_func.py"),
                os.path.join(root_folder, "utils", "data_manager.py"),
                os.path.join(root_folder, "utils", "torchtext_module.py"),
                os.path.join(root_folder, "read2vec", "basic.py"),
                os.path.join(root_folder, "read2vec", "fastDnaEmbed.py"),
                os.path.join(root_folder, "read2vec", "fastTextEmbed.py"),
                os.path.join(root_folder, "read2vec", "transformer.py"),
                os.path.join(root_folder, "read2vec", "SIF.py"),
                os.path.join(root_folder, "read2vec", "read2vec.py"),
                os.path.join(root_folder, "read2genome", "read2genome.py"),
                os.path.join(root_folder, "read2genome", "transformerClassifier.py"),
                os.path.join(root_folder, "read2genome", "h2oModel.py"),
                os.path.join(root_folder, "read2genome", "fastDnaPred.py")]
    for py_file in py_files:
        spark.sparkContext.addFile(py_file)
    return spark


def write_csv_from_spark(df, path_save, sep=",", mode="overwrite"):
    """
    Saving a spark dataframe into one csv file

    :param df: Spark Dataframe
    :param path_save: String, path of the csv file
    :param sep: String, csv separator
    :param mode: String, saving mode
    :return:
    """
    columns = df.columns.copy()
    path_tmp = path_save + "_tmp"
    df.write.csv(path_tmp, sep=sep, mode=mode)
    subprocess.call("echo %s > %s" % (sep.join(columns), path_save), shell=True)
    for file in os.listdir(path_tmp):
        if file.endswith(".csv"):
            subprocess.call("cat %s >> %s" % (os.path.join(path_tmp, file), path_save), shell=True)
    subprocess.call(["rm", "-r", path_tmp])


def file_hdfs_to_rdd(sc, file_hdfs):
    """
    Create a rdd by file in list_files and concatenate all in one rdd.

    :param sc: SparkContext, current spark context
    :param file_hdfs: String, The complete name of the file on hdfs
    :return: Spark RDD, the rdd of the file number num_file in the list_file_hdfs
    """
    return sc.textFile(file_hdfs)


def generate_GloVe_word_count(path_file_name, df_word_count):
    """
    Create a vocab count under the glove format of rdd_word_count.

    :param path_file_name: String, complete file name where are saved the data
    :param df_word_count: pySpark Dataframe, the df after the function df_to_df_word_count
    :return:
    """
    with open(path_file_name, "w") as f:
        for word_count in df_word_count.select(["kmer", "count"]).collect():
            f.write("%s %s\n" % (word_count[0], word_count[1]))
    f.close()


def generate_GloVe_cooccurrence(path_file_name, df_context, df_word_count, mode="hdfs"):
    """
    Create a cooccurrence matrix in binary format. This new file as to be given to
    the glove implementation
    :param path_file_name: String, complete file name where are saved the data
    :param df_context: pySpark DataFrame
    :param df_word_count: pySpark DataFrame, the rdd created by the function rdd_word_count_to_rdd_index
    :return:
    """
    index = df_word_count.select("kmer", "id").rdd.collectAsMap()

    def select_index(x):
        return index[x] if x in index else 0

    # Path until the C script convertor
    path_generate_cooccurrence = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "data_processing/glove")
    c_script = os.path.join("./", path_generate_cooccurrence, "generate_cooccurrence")

    rdd_context = df_context.select("kmer", "context", "count").rdd.map(lambda x: ((select_index(x[0]), select_index(x[1])), x[2]))
    tmp_file = "cooccurrence_glove_tmp"
    if mode == "hdfs":
        rdd_context.saveAsTextFile(tmp_file)
        subprocess.call(["hdfs", "dfs", "-getmerge", tmp_file, tmp_file])
        subprocess.call(["hdfs", "dfs", "-rm", "-r", tmp_file])
        subprocess.call(["sed", "-i", "s/[('),]//g", tmp_file])
        subprocess.call("%s < %s > %s" % (c_script, tmp_file, path_file_name), shell=True)
        subprocess.call(["rm", tmp_file])
    if mode == "local":
        rdd_context = rdd_context.coalesce(1)
        rdd_context.saveAsTextFile(tmp_file)
        subprocess.call(["sed", "-i", "s/[('),]//g", os.path.join(tmp_file, "part-00000")])
        subprocess.call("%s < %s > %s" % (c_script, os.path.join(tmp_file, "part-00000"), path_file_name), shell=True)
        subprocess.call(["rm", "-r", tmp_file])


def read_raw_fastq_file_to_df(spark, path_data, n_sample_load=1., num_partitions=-1, in_memory=False):
    """
    Read raw fastq files saved in a folder and return a dataframe
    Format fastq : id, nucleotide sequence, description, scores
    :param path_data: String, path to the folder containing .fq.gz files
    :param n_sample_load: float, percentage of the rdd taken
    :return: pyspark RDD, element = String = sequence of nucleotides
    """
    if in_memory:
        schema_fasta = T.StructType(
            [T.StructField(read_id_name, T.StringType(), False)])
        df = spark.read.csv(path_data, header=False, schema=schema_fasta)
        if n_sample_load != -1:
            df = df.limit(n_sample_load * 4).persist()  # 4 times because of the additional element of a fasta file
            df.count()
        df = df.withColumn(index_name, F.row_number().over(Window.orderBy(F.monotonically_increasing_id())))
        df_read = df.filter(df.index % 4 == 2).withColumnRenamed(read_id_name, read_name)
        df_id = df.filter(df.index % 4 == 1).withColumnRenamed(read_id_name, read_id_name)
        df_id = df_id.withColumn(pair_name, F.udf(lambda x: x.split('/')[-1], T.StringType())(read_id_name))
        df_read = df_read.withColumn(index_name, F.monotonically_increasing_id())
        df_id = df_id.withColumn(index_name, F.monotonically_increasing_id())
        df_read = df_read.persist()
        count_read = df_read.count()
        df_id = df_id.persist()
        count_id = df_id.count()
        assert count_id == count_read, "The join is impossible due to the difference of rows"
        df = df_id.join(df_read, on=index_name, how="inner").drop(index_name)
        if num_partitions > 0:
            df = df.repartition(num_partitions)
        df = df.filter(df.read_id != "+")
        df = df.persist()
        df.count()
    else:
        schema_fasta = T.StructType(
            [T.StructField(read_id_name, T.StringType(), False), T.StructField(read_name, T.StringType(), False)])
        L_list_tmp_file = []
        str_tmp = "tmp_cleaning_raw_data_"
        for filename in os.listdir(path_data):
            if filename.startswith(str_tmp):
                subprocess.call(["rm", os.path.join(path_data, filename)])
                continue
            tmp_file_name = os.path.join(path_data, str_tmp + filename).replace(".gz", "")
            L_list_tmp_file.append(tmp_file_name)
            cmd = r"""zcat {} | awk 'NR%4==1 || NR%4==2' | awk 'NR%2{{printf "%s,",$0;next;}}1' > {}""".format(
                os.path.join(path_data, filename), tmp_file_name)
            subprocess.call(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        df = spark.read.csv(L_list_tmp_file, header=False, schema=schema_fasta)
        if num_partitions is not None:
            df = df.repartition(num_partitions)
        if n_sample_load > 0:
            df = df.limit(n_sample_load)
        df = df.withColumn(pair_name, F.udf(lambda x: x.split('/')[-1], T.StringType())(read_id_name))
        df = df.persist()
        df.count()
        for filename in L_list_tmp_file:
            subprocess.call(["rm", filename])
    return df


def df_to_df_word_count(df, k, s, num_partitions=None):
    """
    Calculate the occurence of each word (k_mer)

    :param df: pyspark datafraeme, Contains nucléeotides
    :param k: int, number of nucleotides by chain
    :param s: int, The step between each sequences
    :param num_partitions: Int, number of partition of the final rdd
    :return: rdd_word_count : pyspark RDD, element = (String, int) = (k_mer,count)
    """
    read_col = "read"
    kmer_col = "kmer"
    length_col = "length"
    df = df.withColumn(length_col, F.length(df[read_col]))
    df = df.filter(df[length_col] >= k).drop(length_col).persist()
    df.count()
    udfKmer = F.udf(lambda x: transformation_ADN.cut_word(str(x), k, s, True).tolist(), T.ArrayType(T.StringType()))
    df_word_count = df.select(read_col).withColumn(kmer_col, F.explode(udfKmer(df.read))).drop(read_col).groupBy(kmer_col).count()
    df_word_count = df_word_count.repartition(num_partitions, kmer_col) if num_partitions is not None else df_word_count
    return df_word_count


def df_word_add_index(df_word_count, num_partitions=None):
    """
    Create a dataframe that contains the index of each word

    :param spark: SparkSession
    :param df_word_count: pyspark Dataframe, element = (String, int) = (k_mer,count)
    :param num_partitions: Int, number of partition of the final rdd
    :return: pyspark DataFrame, element = (String, int) = (k_mer,index)
    """
    schema = T.StructType([T.StructField("kmer", T.StringType(), False),
                           T.StructField("count", T.LongType(), False),
                           T.StructField("id", T.LongType(), False)])
    df_index = df_word_count.sort("count", ascending=False).rdd.zipWithIndex().map(lambda x: (x[0][0], x[0][1], x[1] + 1)).toDF(schema)
    df_index = df_index.repartition(num_partitions, "id") if num_partitions is not None else df_index
    return df_index


def get_rdd_GC_rate(rdd, index_chain=None, apply_func=None):
    """
    TODO
    :param rdd:
    :param index_chain:
    :param apply_func:
    :return:
    """
    # index_chain indicate if the data is formed as tuple and which indice we have to take
    if index_chain is None:
        if apply_func is None:
            return rdd.map(lambda x: 1. * (x.count("C") + x.count("G")) / len(x)).mean()
        return rdd.map(lambda x: 1. * (apply_func(x).count("C") + apply_func(x).count("G")) / len(apply_func(x))).mean()
    if apply_func is None:
        return rdd.map(lambda x: 1. * (x[index_chain].count("C") + x[index_chain].count("G")) / len(x[index_chain])).mean()
    return rdd.map(lambda x: 1. * (apply_func(x)[index_chain].count("C") + apply_func(x)[index_chain].count("G")) / len(apply_func(x)[index_chain])).mean()


def merge_df_count(df_count1, df_count2, on, num_partitions):
    """
    Merge two dataframe which represent a counter an sum col1 and col2

    :param df_count1: pyspark dataframe, element = ((String, String), int) = ((k_mer,context),count)
    :param df_count2: pyspark dataframe, element = ((String, String), int) = ((k_mer,context),count)
    :param on: List, list of String Name of the columns for the join operation
    :param num_partitions: Int, number of partition of the final rdd
    :return: pyspark dataframe, element = ((String, String), int) = ((k_mer,context),count)
    """
    if df_count1 is None:
        return df_count2
    if df_count2 is None:
        return df_count1
    col1, col2 = "count", "count2"
    df_count2 = df_count2.withColumnRenamed(col1, col2)
    df_count1 = df_count1.join(df_count2, on=on, how="left_outer").fillna(0)
    df_count1 = df_count1.withColumn(col1, sum([df_count1[col1], df_count1[col2]])).drop(col2)
    df_count1 = df_count1.repartition(num_partitions, *on) if num_partitions is not None else df_count1
    return df_count1
