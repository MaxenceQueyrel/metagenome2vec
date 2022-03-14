from read2genome import Read2Genome
import h2o
import pysparkling
import pandas as pd


class H2oModel(Read2Genome):
    def __init__(self, path_read2genome, hc):
        Read2Genome.__init__(self, "h2o_model")
        self.path_read2genome = path_read2genome
        self.model = h2o.load_model(self.path_read2genome)
        self.hc = hc

    def read2genome(self, df):
        """
        transform kmers into a read embedding
        :param df: pyspark DataFrame
        :return: pandas DataFrame with prediction
        """
        if float(pysparkling.__version__[:4]) >= 3.30:
            h2o_frame = self.hc.asH2OFrame(df)
        else:
            h2o_frame = self.hc.as_h2o_frame(df)
        preds = self.model.predict(h2o_frame)
        df_pd = self.df_pred_to_pandas(preds, h2o_frame)
        return df_pd

    def df_pred_to_pandas(self, preds, df):
        pred_name = "predict"
        prob_name = "prob"
        preds = preds.as_data_frame()
        preds[prob_name] = preds[preds.columns.values[1:]].max(axis=1)
        preds = preds[[pred_name, prob_name]]
        df = df[self.getNotFeatures(df.columns)].as_data_frame()
        df = pd.concat([df, preds], axis=1)
        return df
