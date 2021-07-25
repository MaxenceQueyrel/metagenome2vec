import os
import sys
import pyspark.sql.types as T
import pandas as pd
import pysparkling
from pysparkling import H2OContext
import h2o
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.automl import H2OAutoML
from h2o.grid.grid_search import H2OGridSearch

root_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, os.path.join(root_folder, "utils"))
sys.path.insert(0, os.path.join(root_folder, "read2vec"))

import hdfs_functions as hdfs
import parser_creator
import data_manager
import logger
from string_names import *

SEED = 42


def run_read2vec(spark, path_matrix, df_metadata, path_save_read2vec, num_partitions=None, overwrite=False):
    """
    Runs read2vec algorithm on simulated reads and creates/saves the final data frame for read2genome
    :param spark: SQL spark context
    :param path_matrix: str, name of the simulated data to transform
    :param df_metadata: Spark Dataframe, contains the reference for all taxonomy level
    :param path_save_read2vec: String, path where are saved the final dataframe
    :param num_partitions: Number of partition of the dataframe
    :param overwrite: Boolean, if True rewrite a matrix if it exists
    :return:
    """
    if overwrite or not hdfs.dir_exists(path_save_read2vec, mode):
        # Loading data
        log.write("Loading the data set")
        log.write(path_matrix)
        df_simulated = pd.read_csv(path_matrix, sep="\t")
        # Convert tax id name to str because tax_taken is of type list(str)
        log.write(tax_taken)
        df_simulated[tax_id_name] = df_simulated[tax_id_name].apply(str)
        if tax_taken is not None:
            df_simulated = df_simulated[df_simulated[tax_id_name].isin(tax_taken)]
        if n_sample_load > 0 and n_sample_load < len(df_simulated):
            df_simulated = df_simulated.sample(n_sample_load, random_state=SEED)
        log.write("Data loaded, contains %s reads." % (df_simulated.shape[0]))
        # Transform dataset to spark dataframe
        schema = T.StructType([T.StructField(tax_id_name, T.StringType(), False),
                               T.StructField(sim_id_name, T.StringType(), False),
                               T.StructField(prop_name, T.DoubleType(), False),
                               T.StructField(read_name, T.StringType(), True)])
        df_simulated = spark.createDataFrame(df_simulated[[tax_id_name, sim_id_name, prop_name, read_name]], schema=schema)
        # repartitioning
        if num_partitions is not None:
            df_simulated = df_simulated.repartition(num_partitions)
        # Run read2vec
        log.write("Compute")
        log.write("Number of line in df_simulated %s" % df_simulated.count())
        df_simulated = r2v.read2vec(df_simulated, drop_col_name=False)
        # Add taxonomy level
        df_simulated = df_simulated.withColumn(tax_id_name, df_simulated.tax_id.cast(T.StringType()))
        df_simulated = df_simulated.join(df_metadata, on=tax_id_name)
        df_simulated.write.save(path_save_read2vec, mode=saving_mode, format="parquet")
    else:
        log.write("Already computed so do nothing")


def getFeatures(columns):
    """
    :param columns: List of str
    :return:
    """
    features = []
    for col in columns:
        try:
            a = int(col)
            features.append(str(a))
        except:
            continue
    return features


if __name__ == "__main__":

    ####################################
    # ------ Script's Parameters ------#
    ####################################

    parser = parser_creator.ParserCreator()
    args = parser.parser_train_h2o_model()

    read2vec = args.read2vec
    assert len(args.path_data.split(",")) == 2, "You have to give one path for training and another one for validation"
    path_data_train, path_data_valid = args.path_data.split(",")
    path_data_train = path_data_train[:-1] if path_data_train.endswith('/') else path_data_train
    path_data_valid = path_data_valid[:-1] if path_data_valid.endswith('/') else path_data_valid
    path_save = args.path_save
    overwrite = args.overwrite

    k = args.k_mer_size
    id_gpu = [int(x) for x in args.id_gpu.split(',')]
    path_metagenome_word_count = args.path_metagenome_word_count
    path_log = args.path_log
    log_file = args.log_file
    mode = args.mode
    n_sample_load = args.n_sample_load
    saving_mode = "overwrite" if overwrite else None
    num_partitions = args.num_partitions
    y_name = args.tax_level
    path_model = args.path_model
    read2genome_name = args.f_name
    tax_taken = None if (args.tax_taken is None or args.tax_taken == "None") else [str(x) for x in args.tax_taken.split('.')]
    n_tax_taken = len(tax_taken) if isinstance(tax_taken, list) else None
    nfolds = args.nfolds
    machine_learning_algorithm = args.machine_learning_algorithm
    max_models = args.max_models
    path_read2vec = args.path_read2vec
    path_tmp_folder = args.path_tmp_folder
    only_transform = args.only_transform
    path_metadata = args.path_metadata

    log = logger.Logger(path_log, log_file, log_file,
                        variable_time={"k": k,
                                       "read2vec": read2vec},
                        **vars(args))

    ####################################
    # ---------- Spark Conf -----------#
    ####################################

    spark = hdfs.createSparkSession("read2genome")

    #############################
    # --- Preparing read2vec ---#
    #############################

    r2v = data_manager.load_read2vec(read2vec, path_read2vec=path_read2vec, spark=spark,
                                     path_metagenome_word_count=path_metagenome_word_count, k=k, id_gpu=id_gpu,
                                     path_tmp_folder=path_tmp_folder)

    #############################
    #--- load data and model ---#
    #############################

    # Defining a name for the read2genome algorithm

    path_save_read2vec = os.path.join(path_save, read2genome_name)
    # Adding information from the dataset train name
    path_save_read2vec_train = path_save_read2vec + "_" + path_data_train.split("/")[-1]
    path_save_read2vec_valid = path_save_read2vec + "_" + path_data_valid.split("/")[-1]

    df_metadata = df_taxonomy_ref = pd.read_csv(path_metadata).astype(str).rename(columns={ncbi_id_name: tax_id_name})

    if tax_taken is not None:
        path_save_read2vec_valid = path_save_read2vec_valid + "_n_tax_%s" % n_tax_taken
        path_save_read2vec_train = path_save_read2vec_train + "_n_tax_%s" % n_tax_taken
        # Change the list of tax taken at the taxonomy level if for example y_name is at the genus level
        tax_taken = list(df_metadata[df_metadata[y_name].isin(tax_taken)][tax_id_name].drop_duplicates())
    if n_sample_load > 0:
        path_save_read2vec_valid = path_save_read2vec_valid + "_n_sample_%s" % n_sample_load
        path_save_read2vec_train = path_save_read2vec_train + "_n_sample_%s" % n_sample_load

    df_metadata = spark.createDataFrame(df_metadata[[tax_id_name, species_name, genus_name, family_name]])

    #############################
    #--- read to embeddings ----#
    #############################
    log.write("Running read2vec")

    log.writeExecutionTime()
    run_read2vec(spark, path_data_train, df_metadata, path_save_read2vec_train, num_partitions, overwrite)
    log.writeExecutionTime()
    log.writeExecutionTime()
    # the valid dataset is by default reads_genomes_valid
    run_read2vec(spark, path_data_valid, df_metadata, path_save_read2vec_valid, num_partitions, overwrite)
    log.writeExecutionTime()
    log.write("read2vec done !")

    #############################
    #------ running bench ------#
    #############################

    if only_transform:
        sys.exit(0)

    log.write("Preparing data for learning")

    df_train = spark.read.parquet(path_save_read2vec_train)
    if num_partitions is not None:
        df_train = df_train.repartition(num_partitions)

    features = getFeatures(df_train.columns)

    if float(pysparkling.__version__[:4]) >= 3.30:
        hc = H2OContext.getOrCreate()
    else:
        hc = H2OContext.getOrCreate(spark)

    if machine_learning_algorithm == "aml":
        params = None
        max_models = 1 if max_models <= 0 else max_models
        model = H2OAutoML(exclude_algos=["XGBoost", "DRF", "GLM", "GBM", "StackedEnsemble"],
                          max_models=max_models,
                          seed=SEED,
                          nfolds=nfolds,
                          max_runtime_secs=86400,
                          )
    elif machine_learning_algorithm == "gbm":
        params = {'learn_rate': [i * 0.01 for i in range(1, 11)],
                  'ntrees': list(range(10, 60)),
                  'max_depth': list(range(2, 11)),
                  'sample_rate': [i * 0.1 for i in range(5, 11)],
                  'col_sample_rate': [i * 0.1 for i in range(1, 11)]}
        model = H2OGradientBoostingEstimator
    elif machine_learning_algorithm == "glm":
        model = H2OGeneralizedLinearEstimator
        params = {"family": "multinomial",
                  "alpha": ["L1", "L2"],
                  'lambda': [i * 0.01 for i in range(1, 11)]}
    elif machine_learning_algorithm == "dl":
        params = {"epochs": list(range(10, 60)),
                  "hidden": [[100, 100], [50, 200, 50], [400, 300, 200, 100]],
                  "activation": ["Rectifier", "RectifierWithDropout"],
                  'rate': [i * 0.01 for i in range(1, 11)]}
        model = H2ODeepLearningEstimator
    else:  # Random Forest
        params = {'ntrees': list(range(10, 60)),
                  'max_depth': list(range(2, 11)),
                  'sample_rate': [i * 0.1 for i in range(5, 11)]}
        model = H2ORandomForestEstimator

    if max_models > 1 and machine_learning_algorithm != "aml":
        # Search criteria
        search_criteria = {'strategy': 'RandomDiscrete', 'max_models': max_models, 'seed': SEED}

        model = H2OGridSearch(model=model,
                              grid_id='grid_search',
                              hyper_params=params,
                              search_criteria=search_criteria)
    elif machine_learning_algorithm != "aml":
        # Use default values
        model = model(distribution="multinomial")

    log.write("Spark dataframe to h2o frame")
    if float(pysparkling.__version__[:4]) >= 3.30:
        h2o_train = hc.asH2OFrame(df_train, h2oFrameName="train")
    else:
        h2o_train = hc.as_h2o_frame(df_train, framename="train")
    print(h2o_train.columns)
    h2o_train[y_name] = h2o_train[y_name].asfactor()

    log.writeExecutionTime()
    log.write("Training H2O model")
    if max_models > 1 and machine_learning_algorithm != "aml":
        model.train(x=features,
                    y=y_name,
                    training_frame=h2o_train,
                    seed=SEED,
                    nfolds=nfolds)
    elif machine_learning_algorithm == "aml":
        model.train(x=features,
                    y=y_name,
                    training_frame=h2o_train)
    else:  # not AML and not grid search
        df_valid = spark.read.parquet(path_save_read2vec_valid)
        if num_partitions is not None:
            df_valid = df_valid.repartition(num_partitions)
        if float(pysparkling.__version__[:4]) >= 3.30:
            h2o_valid = hc.asH2OFrame(df_valid, h2oFrameName="train")
        else:
            h2o_valid = hc.as_h2o_frame(df_valid, framename="train")
        h2o_valid[y_name] = h2o_valid[y_name].asfactor()
        model.train(x=features,
                    y=y_name,
                    training_frame=h2o_train,
                    validation_frame=h2o_valid)
    log.writeExecutionTime()

    if max_models > 1 and machine_learning_algorithm != "aml":
        model = model.get_grid(sort_by='mean_per_class_accuracy', decreasing=True).models[0]
    elif machine_learning_algorithm == "aml":
        lb = model.leaderboard
        log.write(model.leaderboard.head(rows=lb.nrows))
        model = model.leader
    name_model = model.algo
    name_model = name_model + "_" + read2genome_name
    name_model = name_model + "_" + y_name
    if tax_taken is not None:
        name_model = name_model + "_n_tax_%s" % n_tax_taken
    path_model = os.path.join(path_model, name_model)
    hdfs.create_dir(path_model, mode)

    h2o_name = "model.h2o"
    model.model_id = h2o_name
    h2o.save_model(model, path_model, force=True)

    log.write("H2O model saved !")
    log.writeExecutionTime("read2genome")
    hc.stop()
    spark.stop()
    log.close()
