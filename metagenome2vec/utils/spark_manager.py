import os
from pyspark.sql import types as T
from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark import SparkConf
from pyspark.sql import SparkSession
import subprocess

from metagenome2vec.utils.string_names import *
from metagenome2vec.utils import transformation_ADN

root_folder = os.path.join(os.environ["METAGENOME2VEC_PATH"], "metagenome2vec")


spark_default_conf = {
    "spark.locality.wait": 0,
    "spark.sql.autoBroadcastJoinThreshold": -1,
    "spark.scheduler.minRegisteredResourcesRatio": 1,
    "spark.executor.extraLibraryPath": "$METAGENOME2VEC_PATH/metagenome2vec/utils/transformation_ADN.cpython-38-x86_64-linux-gnu.so",
    "spark.cleaner.referenceTracking": "false",
    "spark.cleaner.referenceTracking.blocking": "false",
    "spark.cleaner.referenceTracking.blocking.shuffle": "false",
    "spark.cleaner.referenceTracking.cleanCheckpoints": "false",
    "spark.rpc.message.maxSize": 1024,
    "num-executors": 3,
    "executor-cores": 5,
    "driver-memory": "10g",
    "executor-memory": "5g",
    "master": "local[*]",
    "spark.network.timeout": 800,
    "spark.driver.memoryOverhead": "5g",
    "spark.executor.memoryOverhead": "5g",
}


def createSparkSession(name_app, **conf):
    """
    Create a Spark Session
    :param name_app: String
    :param conf: dict the spark context conf
    :return: SparkContext
    """
    conf_ = SparkConf()
    conf_.setAppName(name_app)
    conf = {**spark_default_conf, **conf}

    if "master" in conf:
        conf_ = conf_.setMaster(conf["master"])
        del conf["master"]
    if bool(conf):
        conf_ = conf_.setAll(list(conf.items()))
    spark = SparkSession.builder.config(conf=conf_).getOrCreate()
    py_files = [
        os.path.join(os.environ["METAGENOME2VEC_PATH"], m2v_zip_name),
        os.path.join(root_folder, "utils", "file_manager.py"),
        os.path.join(
            root_folder, "utils", "transformation_ADN.cpython-39-x86_64-linux-gnu.so"
        ),
        os.path.join(root_folder, "utils", "stat_func.py"),
        os.path.join(root_folder, "utils", "data_manager.py"),
        os.path.join(root_folder, "read2vec", "basic.py"),
        os.path.join(root_folder, "read2vec", "fastDnaEmbed.py"),
        os.path.join(root_folder, "read2vec", "fastTextEmbed.py"),
        os.path.join(root_folder, "read2vec", "read2vec.py"),
        os.path.join(root_folder, "read2genome", "read2genome.py"),
        os.path.join(root_folder, "read2genome", "fastDnaPred.py"),
    ]
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
            subprocess.call(
                "cat %s >> %s" % (os.path.join(path_tmp, file), path_save), shell=True
            )
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
    path_generate_cooccurrence = os.path.join(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
        "data_processing/glove",
    )
    c_script = os.path.join("./", path_generate_cooccurrence, "generate_cooccurrence")

    rdd_context = df_context.select("kmer", "context", "count").rdd.map(
        lambda x: ((select_index(x[0]), select_index(x[1])), x[2])
    )
    tmp_file = "cooccurrence_glove_tmp"
    if mode == "hdfs":
        rdd_context.saveAsTextFile(tmp_file)
        subprocess.call(["hdfs", "dfs", "-getmerge", tmp_file, tmp_file])
        subprocess.call(["hdfs", "dfs", "-rm", "-r", tmp_file])
        subprocess.call(["sed", "-i", "s/[('),]//g", tmp_file])
        subprocess.call(
            "%s < %s > %s" % (c_script, tmp_file, path_file_name), shell=True
        )
        subprocess.call(["rm", tmp_file])
    if mode == "local":
        rdd_context = rdd_context.coalesce(1)
        rdd_context.saveAsTextFile(tmp_file)
        subprocess.call(
            ["sed", "-i", "s/[('),]//g", os.path.join(tmp_file, "part-00000")]
        )
        subprocess.call(
            "%s < %s > %s"
            % (c_script, os.path.join(tmp_file, "part-00000"), path_file_name),
            shell=True,
        )
        subprocess.call(["rm", "-r", tmp_file])


def read_raw_fastq_file_to_df(
    spark, path_data, n_sample_load=1.0, num_partitions=-1, in_memory=False
):
    """
    Read raw fastq files saved in a folder and return a dataframe
    Format fastq : id, nucleotide sequence, description, scores
    :param path_data: String, path to the folder containing .fq.gz files
    :param n_sample_load: float, percentage of the rdd taken
    :return: pyspark RDD, element = String = sequence of nucleotides
    """
    if in_memory:
        schema_fasta = T.StructType(
            [T.StructField(read_id_name, T.StringType(), False)]
        )
        df = spark.read.csv(path_data, header=False, schema=schema_fasta)
        if n_sample_load != -1:
            df = df.limit(
                n_sample_load * 4
            ).persist()  # 4 times because of the additional element of a fasta file
            df.count()
        df = df.withColumn(
            index_name,
            F.row_number().over(Window.orderBy(F.monotonically_increasing_id())),
        )
        df_read = df.filter(df.index % 4 == 2).withColumnRenamed(
            read_id_name, read_name
        )
        df_id = df.filter(df.index % 4 == 1).withColumnRenamed(
            read_id_name, read_id_name
        )
        df_id = df_id.withColumn(
            pair_name, F.udf(lambda x: x.split("/")[-1], T.StringType())(read_id_name)
        )
        df_read = df_read.withColumn(index_name, F.monotonically_increasing_id())
        df_id = df_id.withColumn(index_name, F.monotonically_increasing_id())
        df_read = df_read.persist()
        count_read = df_read.count()
        df_id = df_id.persist()
        count_id = df_id.count()
        assert (
            count_id == count_read
        ), "The join is impossible due to the difference of rows"
        df = df_id.join(df_read, on=index_name, how="inner").drop(index_name)
        if num_partitions > 0:
            df = df.repartition(num_partitions)
        df = df.filter(df.read_id != "+")
        df = df.persist()
        df.count()
    else:
        schema_fasta = T.StructType(
            [
                T.StructField(read_id_name, T.StringType(), False),
                T.StructField(read_name, T.StringType(), False),
            ]
        )
        L_list_tmp_file = []
        str_tmp = "tmp_cleaning_raw_data_"
        for filename in os.listdir(path_data):
            if filename.startswith(str_tmp):
                os.remove(os.path.join(path_data, filename))
        for filename in os.listdir(path_data):
            tmp_file_name = os.path.join(path_data, str_tmp + filename).replace(
                ".gz", ""
            )
            L_list_tmp_file.append(tmp_file_name)
            cmd = r"""zcat {} | awk 'NR%4==1 || NR%4==2' | awk 'NR%2{{printf "%s,",$0;next;}}1' > {}""".format(
                os.path.join(path_data, filename), tmp_file_name
            )
            subprocess.call(
                cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
            )
        df = spark.read.csv(L_list_tmp_file, header=False, schema=schema_fasta)
        if num_partitions is not None:
            df = df.repartition(num_partitions)
        if n_sample_load > 0:
            df = df.limit(n_sample_load)
        df = df.withColumn(
            pair_name, F.udf(lambda x: x.split("/")[-1], T.StringType())(read_id_name)
        )
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
    udfKmer = F.udf(
        lambda x: transformation_ADN.cut_word(x, k, s, True).tolist(),
        T.ArrayType(T.StringType()),
    )
    df_word_count = (
        df.select(read_col)
        .withColumn(kmer_col, F.explode(udfKmer(df.read)))
        .drop(read_col)
        .groupBy(kmer_col)
        .count()
    )
    df_word_count = (
        df_word_count.repartition(num_partitions, kmer_col)
        if num_partitions is not None
        else df_word_count
    )
    return df_word_count


def df_word_add_index(df_word_count, num_partitions=None):
    """
    Create a dataframe that contains the index of each word

    :param spark: SparkSession
    :param df_word_count: pyspark Dataframe, element = (String, int) = (k_mer,count)
    :param num_partitions: Int, number of partition of the final rdd
    :return: pyspark DataFrame, element = (String, int) = (k_mer,index)
    """
    schema = T.StructType(
        [
            T.StructField("kmer", T.StringType(), False),
            T.StructField("count", T.LongType(), False),
            T.StructField("id", T.LongType(), False),
        ]
    )
    df_index = (
        df_word_count.sort("count", ascending=False)
        .rdd.zipWithIndex()
        .map(lambda x: (x[0][0], x[0][1], x[1] + 1))
        .toDF(schema)
    )
    df_index = (
        df_index.repartition(num_partitions, "id")
        if num_partitions is not None
        else df_index
    )
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
            return rdd.map(
                lambda x: 1.0 * (x.count("C") + x.count("G")) / len(x)
            ).mean()
        return rdd.map(
            lambda x: 1.0
            * (apply_func(x).count("C") + apply_func(x).count("G"))
            / len(apply_func(x))
        ).mean()
    if apply_func is None:
        return rdd.map(
            lambda x: 1.0
            * (x[index_chain].count("C") + x[index_chain].count("G"))
            / len(x[index_chain])
        ).mean()
    return rdd.map(
        lambda x: 1.0
        * (
            apply_func(x)[index_chain].count("C")
            + apply_func(x)[index_chain].count("G")
        )
        / len(apply_func(x)[index_chain])
    ).mean()


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
    df_count1 = df_count1.withColumn(
        col1, sum([df_count1[col1], df_count1[col2]])
    ).drop(col2)
    df_count1 = (
        df_count1.repartition(num_partitions, *on)
        if num_partitions is not None
        else df_count1
    )
    return df_count1
