# -*- coding: latin-1 -*-

import os
import time
import sys
import numpy as np
import pandas as pd
import json
import re
from pyspark.sql import functions as F
from pysparkling import H2OContext
import h2o
import pysparkling
import pickle
root_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, os.path.join(root_folder, "utils"))
sys.path.insert(0, os.path.join(root_folder, "read2vec"))
sys.path.insert(0, os.path.join(root_folder, "read2genome"))
import data_manager
import logger
import parser_creator
import hdfs_functions as hdfs
from string_names import *


def compute(df, r2v, metagenome, target, computation_type, n_sample_load=-1, r2g=None):
    """
    Compute the structuring matrix for one metagenome
    :param df: spark dataframe, contains reads to process
    :param r2v: Read2Vec object, used for the read embeddings
    :param metagenome : String, name of the metagenome
    :param target : String, the vs of the metagenome
    :param computation_type: List, list of int referencing the method
    :param n_sample_load: int, number of element kept in the dataframe
    :param r2g: Read2Genome object, used for the read classification
    :return: x_sil pandas dataframe (one element of the X_sil matrix)
             x_mil pandas dataframe (one element of the X_mil matrix)
    """
    if n_sample_load > 0:
        df = df.limit(n_sample_load)
    # Filter reads too short (lower than k)
    df = df.filter(F.length(F.col(read_name)) > k).persist()
    df.count()
    # read2vec
    log.writeExecutionTime()
    df_embed = r2v.read2vec(df)
    log.writeExecutionTime("read2vec")
    if 0 in computation_type:
        res_sil = df_embed.groupBy().mean().toPandas()
        res_sil.columns = [str(x) for x in range(res_sil.shape[1])]
        res_sil.insert(0, id_fasta_name, metagenome)
        if 1 not in computation_type:
            return res_sil, None
    # READ2GENOME

    if r2g is not None:
        log.writeExecutionTime()
        # read2genome never computed
        if not overwrite and path_folder_save_read2genome is not None and os.path.exists(os.path.join(path_folder_save_read2genome, metagenome)):
            df_pred = pd.read_csv(os.path.join(path_folder_save_read2genome, metagenome))
        else:
            if read2genome == "fastDNA":
                df_pred = r2g.read2genome(df)
            else:
                df_pred = r2g.read2genome(df_embed)
            if path_folder_save_read2genome is not None:
                df_pred.to_csv(os.path.join(path_folder_save_read2genome, metagenome), index=False)
        log.writeExecutionTime("read2genome")
        log.writeExecutionTime()
        if in_memory:  # pandas
            df = df_embed.toPandas()
            del df_embed
            # remove duplicated columns
            for col_name in df_pred.columns.values:
                if col_name in df.columns.values:
                    df = df.drop(columns=col_name)
            df = pd.concat([df, df_pred], axis=1)
            del df_pred
        else:  # h2o
            df = asH2OFrame(df_embed)
            del df_embed
            df_pred = asH2OFrame(spark.createDataFrame(df_pred))
            # remove duplicated columns
            for col_name in df_pred.col_names:
                if col_name in df.col_names:
                    df = df.drop(col_name)
            df = df.concat(df_pred, axis=1)
            del df_pred
            #df = df.as_data_frame()
        log.writeExecutionTime("concatenation embed pred")
        # Remove reads under the threshold
        log.writeExecutionTime()
        if threshold > 0.0:
            i = 1.
            while len(df[df[prob_name] >= threshold / i]) == 0:
                i += 1
            df = df[df[prob_name] >= threshold / i]
        log.writeExecutionTime("filter proba")
        log.writeExecutionTime()
        if paired_prediction:
            def get_pred(prob, pred=None):
                if pred is None:
                    return max([float(x) for x in prob.split(",")]) if "," in prob else float(prob)
                else:
                    return pred.split(",")[np.argmax([float(x) for x in prob.split(",")])] if "," in prob else pred
            read_id_tmp_name = "read_id_tmp"
            if in_memory:
                df[read_id_tmp_name] = df[read_id_name].apply(lambda x: str(x)[:-2] if x.endswith('/1') or x.endswith('/2') else str(x))
                df[[pred_name, prob_name]] = df[[pred_name, prob_name]].astype(str)
                df_gb = df.groupby(read_id_tmp_name).agg({pred_name: ",".join, prob_name: ",".join})
                df_gb[pred_name] = df_gb[[prob_name, pred_name]].apply(lambda x: get_pred(*x), axis=1)
                df_gb[prob_name] = df_gb[prob_name].apply(get_pred)
                del df[pred_name]
                del df[prob_name]
                df = df.merge(df_gb, left_on=read_id_tmp_name, right_on=df_gb.index)
                del df[read_id_tmp_name]
            else:
                col_to_add = df[read_id_name].as_data_frame()
                col_to_add[read_id_tmp_name] = col_to_add[read_id_name].apply(
                    lambda x: hash(str(x)[:-2]) if x.endswith('/1') or x.endswith('/2') else hash(str(x)))
                df = df.concat(asH2OFrame(spark.createDataFrame(col_to_add[[read_id_tmp_name]])))
                col_to_add = df[[read_id_tmp_name, pred_name, prob_name]].as_data_frame()
                col_to_add[[pred_name, prob_name]] = col_to_add[[pred_name, prob_name]].astype(str)
                df_gb = col_to_add.groupby(read_id_tmp_name).agg({pred_name: ",".join, prob_name: ",".join})
                df_gb[pred_name] = df_gb[[prob_name, pred_name]].apply(lambda x: get_pred(*x), axis=1)
                df_gb[prob_name] = df_gb[prob_name].apply(get_pred)
                df = df.drop([pred_name, prob_name])
                df_gb = hc.asH2OFrame(spark.createDataFrame(df_gb.reset_index()))
                df = df.merge(df_gb, all_x=True, by_x=[read_id_tmp_name], by_y=[read_id_tmp_name], method="hash").drop(
                    read_id_tmp_name)
            del df_gb
        log.writeExecutionTime("compute pair prediction")
        log.writeExecutionTime()
        if in_memory:  # pandas
            for col_name in [pair_name, read_name, prob_name]:
                if col_name in df.columns.values:
                    df = df.drop(columns=col_name)
        else:  # h2o
            for col_name in [pair_name, read_name, prob_name]:
                if col_name in df.col_names:
                    df = df.drop(col_name)
        log.writeExecutionTime("drop columns")
    else:
        # random prediction
        if in_memory:  # pandas
            df_embed = df_embed.toPandas()
            df = pd.concat([df_embed, pd.DataFrame(np.random.randint(0, n_instance, df.count()), columns=[pred_name])], axis=1)
        else:
            df_embed = asH2OFrame(df_embed)
            df = df_embed.cbind(h2o.H2OFrame.from_python(np.random.randint(0, n_instance, df.count()), column_names=[pred_name]))
    log.writeExecutionTime()
    if in_memory:  # pandas
        df = df.groupby(pred_name).agg(
            {**{col: 'sum' for col in df.columns.values if col.isdigit()}, **{pred_name: count_name}}).rename(
            columns={pred_name: count_name}).reset_index().rename(columns={pred_name: genome_name})
        df = pd.concat([pd.DataFrame(np.array([[metagenome] * df.shape[0], [target] * df.shape[0]]).T,
                                     columns=[id_fasta_name, group_name]), df], axis=1)
    else:  # h2o
        gb = df.group_by(pred_name)
        df = gb.count().sum().get_frame().rename(columns={"nrow": count_name, pred_name: genome_name})
        df.col_names = [re.sub("^sum_", "", col) for col in df.col_names]
        df = h2o.H2OFrame.from_python(np.array([[metagenome] * df.shape[0], [target] * df.shape[0]]).T,
                                      column_names=[id_fasta_name, group_name]).cbind(df)
        df = df.as_data_frame()
    log.writeExecutionTime("compute groupby")
    if 0 in computation_type and 1 in computation_type:
        return res_sil, df
    return None, df


def metagenome2vec(r2v, spark, computation_type, r2g=None):
    """
    Compute the transformation for each metagenome and save it
    :param spark: SparkSession
    """
    # We compute the first folder containing the first metagenome which will be merged with the
    # result of the other folder to create a big database
    # All computation are saved for each folder
    file_name_res_sil = os.path.join(path_save, "tabular.csv")
    file_name_res_mil = os.path.join(path_save, "mil.csv")
    X_sil = X_mil = None
    if overwrite is False and os.path.exists(file_name_res_sil):
        X_sil = pd.read_csv(file_name_res_sil)
    if overwrite is False and os.path.exists(file_name_res_mil):
        X_mil = pd.read_csv(file_name_res_mil)
    for m, metagenome in enumerate(metadata[id_fasta_name]):
        if m > nb_metagenome:
            break
        if X_sil is not None and \
                X_mil is not None and \
                metagenome in X_sil[id_fasta_name].tolist()  and \
                metagenome in X_mil[id_fasta_name].tolist():
            continue  # already exists
        d = time.time()
        log.write("Iteration %s %s" % (m, metagenome))
        df = spark.read.parquet(os.path.join(path_data, metagenome))
        # Only if we are testing the code
        x_sil, x_mil = compute(df, r2v, metagenome, metadata[group_name].iloc[m], computation_type, n_sample_load, r2g)
        if 0 in computation_type:
            if X_sil is not None:
                X_sil = pd.concat([X_sil, x_sil])
            else:
                X_sil = x_sil.copy()
            X_sil.to_csv(file_name_res_sil, index=False)
        if 1 in computation_type:
            if X_mil is not None:
                X_mil = pd.concat([X_mil, x_mil])
            else:
                X_mil = x_mil.copy()
            X_mil.to_csv(file_name_res_mil, index=False)
        log.write("Duration: %s" % (time.time()-d))
        spark.catalog.clearCache()
        del x_sil, x_mil


def metagenome2vec_one(id_label, r2v, spark, computation_type, r2g=None):
    """
    Compute the transformation for each metagenome and save it
    :param spark: SparkSession
    """
    d = time.time()
    metagenome, group = id_label
    if overwrite is False and os.path.exists(os.path.join(path_save_tmp, metagenome)):
        return
    df = spark.read.parquet(os.path.join(path_data, metagenome))
    # Only if we are testing the code
    x_sil, x_mil = compute(df, r2v, metagenome, group, computation_type, n_sample_load, r2g)
    pickle.dump({"x_sil": x_sil, "x_mil": x_mil}, open(os.path.join(path_save_tmp, metagenome), "wb"))
    log.write("Duration: %s" % (time.time()-d))


def metagenome2vec_merge():
    """
    Merge file computed by metagenome2vec_one
    """
    file_name_res_sil = os.path.join(path_save, "tabular.csv")
    file_name_res_mil = os.path.join(path_save, "mil.csv")
    file_name_res_abundance_table = os.path.join(path_save, "abundance_table.csv")
    L_tmp_file = [f for f in os.listdir(path_save_tmp) if os.path.isfile(os.path.join(path_save_tmp, f))]

    X_sil = list()
    X_mil = list()
    for f in L_tmp_file:
        tmp = pickle.load(open(os.path.join(path_save_tmp, f), "rb"))
        x_sil, x_mil = tmp["x_sil"], tmp["x_mil"]
        if x_sil is not None:
            X_sil.append(x_sil)
        if x_mil is not None:
            X_mil.append(x_mil)
    assert X_sil is None or (X_sil is not None and len(X_sil) == len(L_tmp_file)), "sil: All metagenomes have not been well transformed"
    assert X_mil is None or (X_mil is not None and len(X_mil) == len(L_tmp_file)), "mil: All metagenomes have not been well transformed"
    X_mil = pd.concat(X_mil)
    X_sil = pd.concat(X_sil)
    X_abundance_table = X_mil.pivot_table(columns=genome_name, index=id_fasta_name, values=count_name).fillna(0)
    X_abundance_table = X_abundance_table.div(X_abundance_table.sum(axis=1), axis=0)
    X_sil.to_csv(file_name_res_sil, index=False)
    X_mil.to_csv(file_name_res_mil, index=False)
    X_abundance_table.to_csv(file_name_res_abundance_table)


def compute_cut_matrix_read_embeddings(D_L_metagenome, r2v, pct_cut=0.1, n_cut=10):
    """
    Create the matrix for MIL learning with read embeddings
    :return:
    """
    file_name_res_cut_matrix = os.path.join(path_save, "cut_matrix.csv")
    if not overwrite and os.path.isfile(file_name_res_cut_matrix):
        return

    df_res = None
    for key, L_metagenome in D_L_metagenome.items():
        for metagenome in L_metagenome:
            n_iter = n_cut if key == "to_cut" else 1
            to_cut = True if key == "to_cut" else False
            for j in range(n_iter):
                metagenome_name = metagenome if n_iter == 1 else metagenome + "__%s" % j
                log.write("read embeddings, Metagenome: %s" % metagenome_name)
                d = time.time()
                df = spark.read.parquet(os.path.join(path_data, metagenome))
                df = df.sample(False, pct_cut) if to_cut else df
                df = df.persist()
                df.count()
                x_sil, _ = compute(df, r2v, metagenome_name,
                                   str(metadata[metadata[id_fasta_name] == metagenome_name][group_name]),
                                   computation_type=[0], n_sample_load=n_sample_load)
                if df_res is None:
                    df_res = pd.DataFrame([[x_sil.loc[0, id_fasta_name], to_cut] + x_sil.iloc[0].values[1:].tolist()], columns=[id_fasta_name, "is_cut"] + x_sil.columns.values[1:].tolist())
                else:
                    df_tmp = pd.DataFrame([[x_sil.loc[0, id_fasta_name], to_cut] + x_sil.iloc[0].values[1:].tolist()], columns=[id_fasta_name, "is_cut"] + x_sil.columns.values[1:].tolist())
                    df_res = df_res.append(df_tmp, ignore_index=True)
                log.write("Duration : %s" % (time.time() - d))
    log.write("Saving final table")
    df_res.to_csv(file_name_res_cut_matrix, sep=',', index=False)


if __name__ == "__main__":
    ####################################
    # ------ Script's Parameters ------#
    ####################################
    parser = parser_creator.ParserCreator()
    args = parser.parser_metagenome2vec()
    path_save = args.path_save
    k = args.k_mer_size
    num_partitions = args.num_partitions
    mode = args.mode
    path_data = args.path_data

    log_file = args.log_file
    overwrite = args.overwrite
    saving_mode = "overwrite" if overwrite else None
    path_log = args.path_log
    nb_metagenome = args.nb_metagenome
    name_df_read_embeddings = "df_read_embeddings.parquet"
    read2vec = args.read2vec
    path_metagenome_word_count = args.path_metagenome_word_count
    n_sample_load = args.n_sample_load
    computation_type = [int(x) for x in args.computation_type.split(',') if int(x) in [0, 1, 2, 3]]
    path_model = args.path_model
    path_metagenome_cut_analyse = args.path_metagenome_cut_analyse
    path_folder_save_read2genome = args.path_folder_save_read2genome if args.path_folder_save_read2genome != "None" else None
    if path_folder_save_read2genome is not None:
        hdfs.create_dir(path_folder_save_read2genome, mode="local")
    n_instance = args.n_instance
    threshold = args.thresholds
    path_read2vec = args.path_read2vec
    path_tmp_folder = args.path_tmp_folder
    read2genome = args.read2genome
    in_memory = args.in_memory
    paired_prediction = args.paired_prediction
    path_metadata = None if args.path_metadata == "None" else args.path_metadata
    id_label = None if args.id_label == "None" else args.id_label
    assert all([x not in computation_type for x in [0, 1, 2]]) or path_metadata is not None or id_label is not None, "path_metadata or id_label should be defined"
    path_save_tmp = os.path.join(path_save, "tmp")
    if id_label is not None:
        id_label = id_label.split(',')
        assert len(id_label) == 2, "2 values are mandatory id.fasta and group"
        hdfs.create_dir(path_save_tmp, "local")

    hdfs.create_dir(path_save, mode="local")
    id_gpu = [int(x) for x in args.id_gpu.split(',')]

    log = logger.Logger(path_log, log_file, log_file,
                        variable_time={"k": k, "read2vec": read2vec},
                        **vars(args))

    # Init Spark
    spark = hdfs.createSparkSession("metagenome2vec")
    # Init h2o
    if in_memory is False:
        if float(pysparkling.__version__[:4]) >= 3.30:
            hc = H2OContext.getOrCreate()
            asH2OFrame = hc.asH2OFrame
        else:
            hc = H2OContext.getOrCreate(spark)
            asH2OFrame = hc.as_h2o_frame
    else:
        hc = None

    beg = time.time()

    if path_metadata is not None:
        metadata = pd.read_csv(path_metadata, delimiter=",")[[id_fasta_name, group_name]]

    #############################
    # --- Preparing read2vec ----#
    #############################

    r2v = data_manager.load_read2vec(read2vec, path_read2vec=path_read2vec, spark=spark,
                                     path_metagenome_word_count=path_metagenome_word_count, k=k, id_gpu=id_gpu,
                                     path_tmp_folder=path_tmp_folder)

    ###################################
    # ---- Compute transformation -----#
    # ---- And Saving Data ------------#
    ###################################

    if path_model is not None:
        log.write("Loading read2genome model")
        r2g = data_manager.load_read2genome(read2genome, path_model, hc, path_tmp_folder)
    else:
        r2g = None
    if 0 in computation_type or 1 in computation_type:
        log.write("Begin embeddings matrix structuring")
        if id_label is None:
            metagenome2vec(r2v, spark, computation_type, r2g)
        else:
            metagenome2vec_one(id_label, r2v, spark, computation_type, r2g)
        log.write("End computation")
    elif 2 in computation_type:
        log.write("Begin embeddings cut matrix structuring")
        D_L_metagenome = json.load(open(path_metagenome_cut_analyse, "r"))
        compute_cut_matrix_read_embeddings(D_L_metagenome, r2v)
        log.write("End computation")
    elif 3 in computation_type:
        log.write("Begin merge")
        metagenome2vec_merge()
        log.write("End computation")
    log.write("Total time spending for the script : %s" % (time.time() - beg))
    log.close()

