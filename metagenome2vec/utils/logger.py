# -*- coding: utf-8 -*-

import os
import datetime
import time
import pandas as pd


class Logger:
    def __init__(self, path, file_name="log", name_script="", variable_time={"no_var": "none"},
                 extention="txt", header=True, **kwargs):
        """
        Create one log file and one log file for the time.

        For the second one a file named time_variable1_variable2..._variableN.csv is automatically created
        (separator = ','). If variable_time doesn't change the file is updated

        :param path: String, The complete path where is stored the log files
        :param file_name: String (default = "log"), The name of the log file, it is named file_name_x.extention
            where x is a counter
        :param name_script: String (default = ""), write this name script written at the beginning of the log file
        :param variable_time: dictionary (default = {}), {key="varialbe name" : value="variable value"}, it corresponds
            to the variables for which we want to analyse the execution time. there is one column time by default.
            Each time the program runs the logger will put a new execution time at the expected line.
            If the line with the same variables already exists it will be overwrite
        :param extention: String (default = "txt"), the extension of the file that we want to create
        :param header: Boolean (default = True), write the header in the log file
        :param kwargs: Dictionary, all the parameters of the script, they willl be written at the beginning of the
            log file
        """
        i = 0
        if file_name is None:
            file_name = "log"
        if "." in file_name:
            file_name = file_name.split('.')[0]
        if os.path.exists(path) is False:
            os.makedirs(path)
        while os.path.exists(os.path.join(path, file_name+"_%s.%s" % (i, extention))):
            i += 1
        self.path_log = path
        self.f_log_name = os.path.join(path, file_name+"_%s.%s" % (i, extention))
        self.f_log = open(self.f_log_name, "w+")
        self.take_time = False
        self.beginning_script_time = time.time()
        # Header
        if header:
            self.f_log.write("%s\nDate of the script's run : %s\n\n"
                             % (name_script, datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
        if len(kwargs) > 0:
            self.f_log.write("Parameters :\n")
            for item, key in kwargs.items():
                self.f_log.write("- %s = %s\n" % (item, key))
            self.f_log.write("\n")
        # time file
        self.variable_time = variable_time
        s = "_".join(sorted(self.variable_time.keys()))
        if not os.path.exists(os.path.join(self.path_log, "time_%s.csv" % s)):
            f_time = open(os.path.join(self.path_log, "time_%s.csv" % s), "w+")
            f_time.write(",".join(["time"]+sorted(self.variable_time.keys())))
            f_time.close()

    def writeExecutionTime(self, message=""):
        """
        Write the execution time of an operation.
        The first time writeExecutionTime is run, it keeps the beginning time.
        The second time writeExecutionTime is run it will print the duration in the log file with a message given in
            parameters if defined.

        :param message: String, The message to print in the log file
        :return:
        """
        if not self.take_time:
            self.begin = time.time()
            self.take_time = True
        else:
            self.take_time = False
            self.end = time.time()
            duration = self.end - self.begin
            self.f_log.write(str(datetime.datetime.now()) + ": \n")
            if message == "":
                self.f_log.write("Execution time : %ss\n\n" % duration)
            else:
                self.f_log.write("Execution time, %s : %ss\n\n" % (message, duration))
            self.snap()

    def writeSizeSparkRDD(self, rdd=None, n_line=None, name_rdd="", message=""):
        """
        Write the size of a Spark RDD (write the number of elements)

        :param rdd: Spark RDD (default=None)
        :param n_line: int (default=None), in order to avoid a big operation we can compute the number of line
            before and just give the number to the function which will write it
        :param name_rdd: String (default=""), the name of the RDD to write in the log files
        :param message: String (default=""), a log message for the operation
        :return:
        """
        assert rdd is not None or n_line is not None, "rdd or n_line have to be defined"
        if rdd is not None:
            n_line = rdd.count()
        if message != "":
            self.f_log.write("%s\n" % message)
        self.f_log.write("The RDD %s contains %s lines\n\n" % (name_rdd, n_line))
                
    def writeSizeSparkDataFrame(self, df=None, n_line=None, n_col=-1, name_df="", message=""):
        """
        Write the size of a Spark DataFrame (write the number of lines and columns)

        :param df: Spark DataFrame (default=None)
        :param n_line: int (default=None), in order to avoid a big operation we can compute the number of line before
            and juste give the number to function which will write it
        :param n_col: int (default=-1), we can also pass the number of columns
        :param name_df: String (default=""), the name of the DataFrame to write in the log file
        :param message: String (default=""), a log message for the operation
        :return:
        """
        assert df is not None or n_line is not None, "df or n_line have to be defined"
        if df is not None:
            n_line = df.count()
            n_col = len(df.columns)
        if message != "":
            self.f_log.write("%s\n" % message)
        self.f_log.write("The DataFrame %s contains %s lines and %s columns\n\n" % (name_df, n_line, n_col))
    
    def writeSizePandasDataFrame(self, df, name_df="", message=""):
        """
        Write the size of a Pandas DataFrame (write the number of lines and columns)

        :param df: Pandas DataFrame
        :param name_df: String (default=""), the name of the DataFrame to write in the log file
        :param message: String (default=""), a log message for the operation
        :return:
        """
        n_line = df.shape[0]
        n_col = df.shape[1]
        if message != "":
            self.f_log.write("%s\n" % message)
        self.f_log.write("The DataFrame %s contains %s lines and %s columns\n\n" % (name_df, n_line, n_col))
    
    def write(self, message, jump_line=True, snap=True, print_time=True):
        """
        Write the log message in f_log

        :param message: String, a log message
        :param jump_line: Boolean (default=True), jump one line between each message
        :param snap: Boolean (default=True), if True close and re open the log file
        :param print_time: Boolean (default=True), print the current time of the log
        :return:
        """
        if print_time:
            self.f_log.write(str(datetime.datetime.now()) + ": \n")
        if jump_line:
            self.f_log.write("%s\n\n" % message)
        else:
            self.f_log.write("%s\n" % message)
        if snap:
            self.snap()

    def snap(self):
        """
        Just close and reopen self.f_log in order to we can read what has been written during the execution
        :return:
        """
        self.f_log.close()
        self.f_log = open(self.f_log_name, 'a')

    def close(self, footer=True):
        """
        Function that close the log file and calcul the total execution time.

        :param footer: Boolean (default=True), write the footer in the log file
        :return:
        """
        script_end_time = time.time() - self.beginning_script_time
        if footer:
            self.f_log.write("Total time execution of the script : %s\n" % script_end_time)
            self.f_log.write("Date of the script's end running : %s\n\n" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
        self.f_log.close()
        # for the log time file
        s = "_".join(sorted(self.variable_time.keys()))
        df_time_log = pd.read_csv(os.path.join(self.path_log, "time_%s.csv" % s), dtype=str)
        #s_filter = ""
        #for k, v in self.variable_time.items():
        #    if s_filter == "":
        #        s_filter += "(df_time_log['%s']=='%s')" % (k, v)
        #    else:
        #        s_filter += " & (df_time_log['%s']=='%s')" % (k, v)
        #row_to_del = list(df_time_log[eval(s_filter)].index)
        #df_time_log = df_time_log.drop(row_to_del)
        df_time_log.loc[len(df_time_log)] = [str(script_end_time)]+[self.variable_time[k] for k in sorted(self.variable_time.keys())]
        df_time_log.to_csv(os.path.join(self.path_log, "time_%s.csv" % s), index=False)
