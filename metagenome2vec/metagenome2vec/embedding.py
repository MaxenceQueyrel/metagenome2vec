import os
import time
import numpy as np
import pandas as pd
import re
from pyspark.sql import functions as F
import h2o
import pickle
import logging

from metagenome2vec.utils.string_names import *


def compute(df, r2v, metagenome, target, k_mer_size, spark, hc, computation_type, n_sample_load=-1, r2g=None, path_folder_save_read2genome=None,
    overwrite=True, in_memory=False, read2genome_algo="fastdna", threshold=0., paired_prediction=True):
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
    # Filter reads too short (lower than k_mer_size)
    df = df.filter(F.length(F.col(read_name)) > k_mer_size).persist()
    df.count()
    # read2vec
    logging.info("Begin read2vec")
    df_embed = r2v.read2vec(df)
    logging.info("End read2vec")
    if 0 in computation_type:
        res_sil = df_embed.groupBy().mean().toPandas()
        res_sil.columns = [str(x) for x in range(res_sil.shape[1])]
        res_sil.insert(0, id_fasta_name, metagenome)
        if 1 not in computation_type:
            return res_sil, None
    # READ2GENOME

    if r2g is not None:
        logging.info("Begin read2genome")
        # read2genome never computed
        if not overwrite and path_folder_save_read2genome is not None and os.path.exists(os.path.join(path_folder_save_read2genome, metagenome)):
            df_pred = pd.read_csv(os.path.join(path_folder_save_read2genome, metagenome))
        else:
            if read2genome_algo == "fastdna":
                df_pred = r2g.read2genome(df)
            else:
                df_pred = r2g.read2genome(df_embed)
            if path_folder_save_read2genome is not None:
                df_pred.to_csv(os.path.join(path_folder_save_read2genome, metagenome), index=False)
        logging.info("End read2genome")

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
            df = hc.asH2OFrame(df_embed)
            del df_embed
            df_pred = hc.asH2OFrame(spark.createDataFrame(df_pred))
            # remove duplicated columns
            for col_name in df_pred.col_names:
                if col_name in df.col_names:
                    df = df.drop(col_name)
            df = df.concat(df_pred, axis=1)
            del df_pred
            #df = df.as_data_frame()
            logging.info("concatenation embed pred")
        # Remove reads under the threshold
        logging.info("Begin filter proba")
        if threshold > 0.0:
            i = 1.
            while len(df[df[prob_name] >= threshold / i]) == 0:
                i += 1
            df = df[df[prob_name] >= threshold / i]
        logging.info("End filter proba")
        logging.info("Begin compute pair prediction")
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
                df = df.concat(hc.asH2OFrame(spark.createDataFrame(col_to_add[[read_id_tmp_name]])))
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
        logging.info("End compute pair prediction")
        logging.info("Begin drop columns")
        if in_memory:  # pandas
            for col_name in [pair_name, read_name, prob_name]:
                if col_name in df.columns.values:
                    df = df.drop(columns=col_name)
        else:  # h2o
            for col_name in [pair_name, read_name, prob_name]:
                if col_name in df.col_names:
                    df = df.drop(col_name)
        logging.info("End drop columns")
    else:
        # random prediction
        if in_memory:  # pandas
            df_embed = df_embed.toPandas()
            df = pd.concat([df_embed, pd.DataFrame(np.random.randint(0, n_instance, df.count()), columns=[pred_name])], axis=1)
        else:
            df_embed = hc.asH2OFrame(df_embed)
            df = df_embed.cbind(h2o.H2OFrame.from_python(np.random.randint(0, n_instance, df.count()), column_names=[pred_name]))
    logging.info("Begin compute groupby")
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
    logging.info("End compute groupby")
    if 0 in computation_type and 1 in computation_type:
        return res_sil, df
    return None, df


def metagenome2vec(path_data, id_label, r2v, spark, n_sample_load, computation_type, path_save_tmp, r2g=None, overwrite=False):
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
    logging.info("Duration: %s" % (time.time()-d))


def metagenome2vec_merge():
    """
    Merge file computed by metagenome2vec
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
                logging.info("read embeddings, Metagenome: %s" % metagenome_name)
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
                logging.info("Duration : %s" % (time.time() - d))
    logging.info("Saving final table")
    df_res.to_csv(file_name_res_cut_matrix, sep=',', index=False)

