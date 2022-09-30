import os
import argparse
import json
import logging
import copy
from metagenome2vec.utils import spark_manager
from metagenome2vec.data_processing.metagenome import preprocess_metagenomic_data, bok_split, bok_merge
from metagenome2vec.data_processing.dowload_metagenomic_data import download_from_tsv_file
from metagenome2vec.data_processing.simulation import *
from metagenome2vec.read2genome.fastDnaPred import FastDnaPred

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

class ParserCreator(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser('Main command')
        self.subparsers = self.parser.add_subparsers(help="test", dest="command")
        self.D_parser = {}
        self.D_parser["-k"] = {"name": "--k_mer_size", "arg": {"metavar": "k_mer_size",
                                                               "type": int,
                                                               "required": True,
                                                               "help": "Size of the k_mer, number of nucleotides by chain for the current word"}}
        self.D_parser["-w"] = {"name": "--window", "arg": {"metavar": "window",
                                                           "type": int,
                                                           "default": 1,
                                                           "help": "Number of chains preceding and following the current one to consider during the learning"}}
        self.D_parser["-s"] = {"name": "--step", "arg": {"metavar": "step",
                                                         "type": int,
                                                         "default": 1,
                                                         "help": "The number of nucleotides that separate each k_mer"}}
        self.D_parser["-p"] = {"name": "--padding", "arg": {"metavar": "padding",
                                                            "type": int,
                                                            "default": 1,
                                                            "help": "The number of nucleotides that separate the k_mer and its context (could be negative => overlap"}}
        self.D_parser["-lf"] = {"name": "--log_file", "arg": {"metavar": "log_file",
                                                              "type": str,
                                                              "default": "log_file.txt",
                                                              "help": "Name of the log file"}}
        self.D_parser["-pg"] = {"name": "--path_log", "arg": {"metavar": "path_log",
                                                              "type": str,
                                                              "default": "./",
                                                              "help": "local path where is stored the log file"}}
        self.D_parser["-pl"] = {"name": "--path_learning", "arg": {"metavar": "path_learning",
                                                                   "type": str,
                                                                   "default": "hdfs://ma-1-1-t630.infiniband:8020/user/mqueyrel/deepGene/data/learning/",
                                                                   "help": "Path where are saved the structuring matrix."}}
        self.D_parser["-pa"] = {"name": "--path_analysis", "arg": {"metavar": "path_analyses",
                                                                   "type": str,
                                                                   "default": "/data/projects/deepgene/analyses/",
                                                                   "help": "Path where are saved the data to analyse."}}
        self.D_parser["-pm"] = {"name": "--path_model", "arg": {"metavar": "path_model",
                                                                "type": str,
                                                                "default": "./",
                                                                "help": "Path where are stored the trained machine learning models"}}
        self.D_parser["-pd"] = {"name": "--path_data", "arg": {"metavar": "path_data",
                                                               "type": str,
                                                               "required": True,
                                                               "help": "Path to the data."}}
        self.D_parser["-ps"] = {"name": "--path_save", "arg": {"metavar": "path_save",
                                                               "type": str,
                                                               "required": True,
                                                               "help": "Path where will be saved the data."}}
        self.D_parser["-pkc"] = {"name": "--path_kmer_count", "arg": {"metavar": "path_kmer_count",
                                                                      "type": str,
                                                                      "default": None,
                                                                      "help": "Path to the kmer count"}}
        self.D_parser["-pmd"] = {"name": "--path_metadata", "arg": {"metavar": "path_metadata",
                                                                    "type": str,
                                                                    "required": True,
                                                                    "help": "Absolute path to the metadata file"}}
        self.D_parser["-pfq"] = {"name": "--path_fastq_file", "arg": {"metavar": "path_fastq_file",
                                                                      "type": str,
                                                                      "required": True,
                                                                      "help": "Absolute path to the fastq file from the simulation"}}
        self.D_parser["-pmf"] = {"name": "--path_mapping_file", "arg": {"metavar": "path_mapping_file",
                                                            "type": str,
                                                            "required": True,
                                                            "help": "Absolute path to the read mapping file from the simulation"}}
        self.D_parser["-f"] = {"name": "--f_name", "arg": {"metavar": "f_name",
                                                           "type": str,
                                                           "default": None,
                                                           "help": "Full path name of the file to structure, if None the whole dataset is stuctured"}}
        self.D_parser["-fl"] = {"name": "--list_file", "arg": {"metavar": "list_file",
                                                               "type": str,
                                                               "default": None,
                                                               "help": "Comma separated string with file name"}}
        self.D_parser["-dn"] = {"name": "--dataset_name", "arg": {"metavar": "dataset_name",
                                                                  "type": str,
                                                                  "default": "dataset",
                                                                  "help": "If f_name is none this is the name given to the files computed and saved"}}
        self.D_parser["-V"] = {"name": "--vocabulary_size", "arg": {"metavar": "vocabulary_size",
                                                                    "type": int,
                                                                    "default": 4 ** 6,
                                                                    "help": "Number of words/chains considered (more frequent)"}}
        self.D_parser["-ct"] = {"name": "--computation_type", "arg": {"metavar": "computation_type",
                                                                      "type": int,
                                                                      "choices": [0, 1, 2],
                                                                      "default": 0,
                                                                      "help": "0 create both heat map and the structuration learning / 1 create only the structuration / 2 create only the heat map"}}
        self.D_parser["-o"] = {"name": "--overwrite", "arg": {"action": "store_true",
                                                              "help": "If true, replace the previous file if exists else do nothing"}}
        self.D_parser["-mo"] = {"name": "--mode", "arg": {"metavar": "mode",
                                                          "type": str,
                                                          "choices": ["hdfs", "local", "s3"],
                                                          "default": "local",
                                                          "help": "'hdfs', 'local', 's3' : correspond to the file system we use"}}
        self.D_parser["-pas"] = {"name": "--parameter_structu", "arg": {"metavar": "parameter_structu",
                                                                        "type": str,
                                                                        "required": True,
                                                                        "help": "The folder containing the file structured with these parameters"}}
        self.D_parser["-pal"] = {"name": "--parameter_learning", "arg": {"metavar": "parameter_learning",
                                                                         "type": str,
                                                                         "required": True,
                                                                         "help": "The parameters used during the embeddings learning."}}
        self.D_parser["-rl"] = {"name": "--ratio_learning", "arg": {"metavar": "ratio_learning",
                                                                    "type": float,
                                                                    "default": 1.,
                                                                    "help": "Take rl percent of the dataframe to run kmer2vec"}}
        self.D_parser["-nsl"] = {"name": "--n_sample_load", "arg": {"metavar": "n_sample_load",
                                                                    "type": int,
                                                                    "default": 1e7,
                                                                    "help": "Number of sampled load in memory, if -1 load all the samples."}}
        self.D_parser["-B"] = {"name": "--batch_size", "arg": {"metavar": "batch_size",
                                                               "type": int,
                                                               "default": 64,
                                                               "help": "Number of elements passed to the learning"}}
        self.D_parser["-E"] = {"name": "--embedding_size", "arg": {"metavar": "embeddings_size",
                                                                   "type": int,
                                                                   "default": 300,
                                                                   "help": "Dimension of the embedding vector"}}
        self.D_parser["-H"] = {"name": "--hidden_size", "arg": {"metavar": "hidden_size",
                                                                "type": int,
                                                                "default": 200,
                                                                "help": "Hidden dimension of the neural network"}}
        self.D_parser["-S"] = {"name": "--n_steps", "arg": {"metavar": "n_steps",
                                                            "type": int,
                                                            "default": 50001,
                                                            "help": "Number of steps during learning"}}
        self.D_parser["-I"] = {"name": "--n_iterations", "arg": {"metavar": "n_iterations",
                                                                 "type": int,
                                                                 "default": -1,
                                                                 "help": "Number of iterations in one step during learning."}}
        self.D_parser["-R"] = {"name": "--learning_rate", "arg": {"metavar": "learning_rate",
                                                                  "type": float,
                                                                  "default": 1.,
                                                                  "help": "Learning rate for the gradient descent"}}
        self.D_parser["-VS"] = {"name": "--valid_size", "arg": {"metavar": "valid_size",
                                                                "type": float,
                                                                "default": 8,
                                                                "help": "Random set for evaluation"}}
        self.D_parser["-vs"] = {"name": "--valid_size", "arg": {"metavar": "valid_size",
                                                                "type": float,
                                                                "default": None,
                                                                "help": "The percentage amount of data in the validation set"}}
        self.D_parser["-VW"] = {"name": "--valid_window", "arg": {"metavar": "valid_window",
                                                                  "type": int,
                                                                  "default": 100,
                                                                  "help": "(kmer2vec) Select VS examples for the validation in the top VW of the distribution"}}
        self.D_parser["-NS"] = {"name": "--num_sampled", "arg": {"metavar": "num_sampled",
                                                                 "type": int,
                                                                 "default": 8,
                                                                 "help": "(kmer2vec) Number of negative examples to sample"}}
        self.D_parser["-NL"] = {"name": "--n_compute_loss", "arg": {"metavar": "n_compute_loss",
                                                                    "type": int,
                                                                    "default": 10,
                                                                    "help": "Number of time you want to calculate the loss"}}
        self.D_parser["-NSIM"] = {"name": "--n_show_similarity", "arg": {"metavar": "n_show_similarity",
                                                                         "type": int,
                                                                         "default": 3,
                                                                         "help": "(kmer2vec) Number of time you want to calculate the similarity in the neural network"}}
        self.D_parser["-cl"] = {"name": "--continue_learning", "arg": {"action": "store_true",
                                                                       "help": "If True restore the previous graph / session and continue the learning from this point"}}
        self.D_parser["-m"] = {"name": "--method", "arg": {"metavar": "method",
                                                           "type": str,
                                                           "default": "normal",
                                                           "help": "The method used to perform"}}
        self.D_parser["-kea"] = {"name": "--kmer_embeddings_algorithm", "arg": {"metavar": "kmer_embeddings_algorithm",
                                                                                "type": str,
                                                                                "default": "fasttext",
                                                                                "help": "Name of the algorithm used for the embedding"}}
        self.D_parser["-if"] = {"name": "--is_file", "arg": {"action": "store_true",
                                                             "help": "If true, sample only a file else a folder."}}
        self.D_parser["-vfn"] = {"name": "--vocabulary_file_name", "arg": {"metavar": "vocabulary_file_name",
                                                                           "type": str,
                                                                           "default": "vocab.txt",
                                                                           "help": "(glove) This is the name of the word count / vocabulary file."}}
        self.D_parser["-cfn"] = {"name": "--cooccurrence_file_name", "arg": {"metavar": "cooccurrence_file_name",
                                                                             "type": str,
                                                                             "default": "cooccurrence.bin",
                                                                             "help": "(glove) This is the name of the cooccurrence file."}}
        self.D_parser["-cfn"] = {"name": "--cooccurrence_file_name", "arg": {"metavar": "cooccurrence_file_name",
                                                                             "type": str,
                                                                             "default": "cooccurrence.bin",
                                                                             "help": "(glove) This is the name of the cooccurrence file."}}
        self.D_parser["-iuk"] = {"name": "--include_unk_kmer", "arg": {"action": "store_true",
                                                                       "help": "(structuring classif) If true, make structuring with unk kmer else avoid them"}}
        self.D_parser["-fn"] = {"name": "--file_name", "arg": {"metavar": "file_name",
                                                               "type": str,
                                                               "required": True,
                                                               "help": "The name given to the res file"}}
        self.D_parser["-X"] = {"name": "--x_max", "arg": {"metavar": "x_max",
                                                          "type": int,
                                                          "default": 10,
                                                          "help": "(glove) Threashold for extremely common word pairs"}}
        self.D_parser["-lm"] = {"name": "--language_modeling", "arg": {"action": "store_true",
                                                                       "help": "Tells  if the trasnformer train like a language modeling model or not"}}
        self.D_parser["-mng"] = {"name": "--min_ngram", "arg": {"metavar": "min_ngram",
                                                                "type": int,
                                                                "default": 3,
                                                                "help": "Minimum size of ngram"}}
        self.D_parser["-Mng"] = {"name": "--max_ngram", "arg": {"metavar": "max_ngram",
                                                                "type": int,
                                                                "default": 6,
                                                                "help": "Maximuml size of ngram"}}
        self.D_parser["-ng"] = {"name": "--nb_metagenome", "arg": {"metavar": "nb_metagenome",
                                                                   "type": int,
                                                                   "default": 10000,
                                                                   "help": "Number of metagenome"}}
        self.D_parser["-nsc"] = {"name": "--n_sample_by_class", "arg": {"metavar": "n_sample_by_class",
                                                                        "type": int,
                                                                        "default": 10000,
                                                                        "help": "Number of samples by class when structuring simulation data."}}
        self.D_parser["-nc"] = {"name": "--n_cpus", "arg": {"metavar": "n_cpus",
                                                            "type": int,
                                                            "default": 16,
                                                            "help": "Number of process used"}}
        self.D_parser["-itf"] = {"name": "--index_tmp_file", "arg": {"metavar": "index_tmp_file",
                                                                     "type": int,
                                                                     "default": -1,
                                                                     "help": "if stm = 2 then it will start from the index given for the next temporary matrix"}}
        self.D_parser["-isi"] = {"name": "--index_sample_id", "arg": {"metavar": "index_sample_id",
                                                                     "type": int,
                                                                     "default": 1,
                                                                     "help": "Index in the tsv file of the column containing the sample ids."}}
        self.D_parser["-iu"] = {"name": "--index_url", "arg": {"metavar": "index_url",
                                                                "type": int,
                                                                "default": 10,
                                                                "help": "Index in the tsv file of the column containing the sample url."}}
        self.D_parser["-np"] = {"name": "--num_partitions", "arg": {"metavar": "num_partitions",
                                                                    "type": int,
                                                                    "default": 16,
                                                                    "help": "Number of partitions of the rdd file"}}
        self.D_parser["-ns"] = {"name": "--nb_sequences_by_metagenome", "arg": {"metavar": "nb_sequences_by_metagenome",
                                                                                "type": int,
                                                                                "required": True,
                                                                                "help": "Number of sequences by metagenome"}}
        self.D_parser["-MIL"] = {"name": "--multi_instance_layer", "arg": {"metavar": "multi_instance_layer",
                                                                           "type": str,
                                                                           "choices": ["sum", "max", "attention"],
                                                                           "default": "sum",
                                                                           "help": "Define the type of mil layer"}}
        self.D_parser["-TS"] = {"name": "--test_size", "arg": {"metavar": "test_size",
                                                               "type": float,
                                                               "default": 0.2,
                                                               "help": "percentage of test data"}}
        self.D_parser["-ig"] = {"name": "--id_gpu", "arg": {"metavar": "id_gpu",
                                                            "type": str,
                                                            "default": "-1",
                                                            "help": "Comma separated string: Index of the gpus we want to use. If -1 use cpu"}}
        self.D_parser["-nb"] = {"name": "--n_batch", "arg": {"metavar": "n_batch",
                                                             "type": int,
                                                             "default": 1e5,
                                                             "help": "Number of batchs generated to gain in computation time. If too big can raise an OOM"}}
        self.D_parser["-ni"] = {"name": "--n_instance", "arg": {"metavar": "n_instance",
                                                                "type": int,
                                                                "default": 1e5,
                                                                "help": "Number of instance in a set"}}
        self.D_parser["-rv"] = {"name": "--read2vec", "arg": {"metavar": "read2vec",
                                                              "type": str,
                                                              "required": True,
                                                              "choices": ["fastDNA", "fastText", "basic",
                                                                          "transformer"],
                                                              "help": "The read embeddings algorithm used."}}
        self.D_parser["-rg"] = {"name": "--read2genome", "arg": {"metavar": "read2genome",
                                                                 "type": str,
                                                                 "required": True,
                                                                 "choices": ["fastDNA", "h2oModel", "transformer"],
                                                                 "help": "The read2genome algorithm used, fastDNA or h2oModel"}}
        self.D_parser["-nri"] = {"name": "--n_reads_instance", "arg": {"metavar": "n_reads_instance",
                                                                       "type": int,
                                                                       "default": 1e5,
                                                                       "help": "Number of reads in one instance of a bag for read embeddings computation."}}

        self.D_parser["-D"] = {"name": "--weight_decay", "arg": {"metavar": "weight_decay",
                                                                 "type": float,
                                                                 "default": 1e-5,
                                                                 "help": "Decay for L2 normalization"}}
        self.D_parser["-ca"] = {"name": "--catalog", "arg": {"metavar": "catalog",
                                                             "type": str,
                                                             "required": True,
                                                             "help": "Name of the catalog used"}}
        self.D_parser["-owc"] = {"name": "--only_word_count", "arg": {"action": "store_true",
                                                                      "help": "Compute only the word count matrix"}}
        self.D_parser["-prt"] = {"name": "--path_read_transformed", "arg": {"metavar": "path_read_transformed",
                                                                            "type": str,
                                                                            "required": True,
                                                                            "help": "Path where are saved the metagenomes with reads transform into embeddings"}}
        self.D_parser["-nco"] = {"name": "--nb_cutoffs", "arg": {"metavar": "nb_cutoffs",
                                                                 "type": int,
                                                                 "default": -1,
                                                                 "help": "Number of cut off for the adaptive softmax"}}
        self.D_parser["-pmwc"] = {"name": "--path_metagenome_word_count",
                                  "arg": {"metavar": "path_metagenome_word_count",
                                          "type": str,
                                          "default": None,
                                          "help": "Complet path to the metagenome word count"}}
        self.D_parser["-pgd"] = {"name": "--path_genome_dist", "arg": {"metavar": "path_genome_dist",
                                                                       "type": str,
                                                                       "default": None,
                                                                       "help": "Path to the genome distance matrix computed by https://gitlab.pasteur.fr/GIPhy/JolyTree/-/blob/master/README.md"}}
        self.D_parser["-Ml"] = {"name": "--max_length", "arg": {"metavar": "max_length",
                                                                "type": int,
                                                                "default": 20,
                                                                "help": "Maximum size of read proceeded by the algorithm"}}
        self.D_parser["-sc"] = {"name": "--spark_conf", "arg": {"metavar": "spark_conf",
                                                                "type": json.loads,
                                                                "default": {},
                                                                "help": "Dict to set the spark conf (avoid to use spark-submit)"}}
        self.D_parser["-tl"] = {"name": "--tax_level", "arg": {"metavar": "tax_level",
                                                               "type": str,
                                                               "choices": ["tax_id", "species", "genus", "family"],
                                                               "default": "species",
                                                               "help": "Determine the taxonomy level considered"}}
        self.D_parser["-T"] = {"name": "--thresholds", "arg": {"metavar": "thresholds",
                                                               "type": str,
                                                               "default": "0.5",
                                                               "help": "Coma separated thresholds"}}
        self.D_parser["-ba"] = {"name": "--balance", "arg": {"metavar": "balance",
                                                             "type": str,
                                                             "choices": [None, "under", "over", "both"],
                                                             "default": None,
                                                             "help": "Tell the method to balance the data"}}
        self.D_parser["-tt"] = {"name": "--tax_taken", "arg": {"metavar": "tax_taken",
                                                               "type": str,
                                                               "default": None,
                                                               "help": "Point separated string: Index of the taxa"}}
        self.D_parser["-nf"] = {"name": "--nfolds", "arg": {"metavar": "nfolds",
                                                            "type": int,
                                                            "default": 0,
                                                            "help": "Number of folds for the grid search"}}
        self.D_parser["-mla"] = {"name": "--machine_learning_algorithm",
                                 "arg": {"metavar": "machine_learning_algorithm",
                                         "type": str,
                                         "default": "gmb",
                                         "choices": ["gbm", "dl", "rf", "glm", "aml"],
                                         "help": "Machine learning algorithm used"}}
        self.D_parser["-mm"] = {"name": "--max_models", "arg": {"metavar": "max_models",
                                                                "type": int,
                                                                "default": 1,
                                                                "help": "Number of models trained in the random grid search"}}
        self.D_parser["-prv"] = {"name": "--path_read2vec", "arg": {"metavar": "path_read2vec",
                                                                    "type": str,
                                                                    "default": None,
                                                                    "help": "Complete path to read2vec model"}}
        self.D_parser["-prg"] = {"name": "--path_read2genome", "arg": {"metavar": "path_read2genome",
                                                                       "type": str,
                                                                       "default": None,
                                                                       "help": "Complete path to read2vec model"}}
        self.D_parser["-pmca"] = {"name": "--path_metagenome_cut_analyse",
                                  "arg": {"metavar": "path_metagenome_cut_analyse",
                                          "type": str,
                                          "default": None,
                                          "help": "Complete path to the json file containing two keys 'to_cut' and 'no_cut' with values as a list of metagenome names"}}
        self.D_parser["-pkv"] = {"name": "--path_kmer2vec", "arg": {"metavar": "path_kmer2vec",
                                                                    "type": str,
                                                                    "default": None,
                                                                    "help": "Complete path to kmer2vec model"}}
        self.D_parser["-pt"] = {"name": "--path_tmp_folder", "arg": {"metavar": "path_tmp_folder",
                                                                     "type": str,
                                                                     "default": None,
                                                                     "help": "Complete path to the tmp folder used for the script"}}
        self.D_parser["-ot"] = {"name": "--only_transform", "arg": {"action": "store_true",
                                                                    "help": "If True just compute the reads transforming not training"}}
        self.D_parser["-DO"] = {"name": "--dropout", "arg": {"metavar": "dropout",
                                                             "type": float,
                                                             "default": 0.2,
                                                             "help": "Dropout value"}}
        self.D_parser["-DS"] = {"name": "--deepsets_struct", "arg": {"metavar": "deepsets_struct",
                                                                     "type": str,
                                                                     "default": "200,100,1,1",
                                                                     "help": "Comma separated string, first value is the number of hidden layer for phi network, second number of hidden layer for rho network, third number of layers for phi network, fourth number of layers for rho network"}}
        self.D_parser["-DV"] = {"name": "--vae_struct", "arg": {"metavar": "vae_struct",
                                                                "type": str,
                                                                "default": "40,4,1",
                                                                "help": "Comma separated string, first value is the number of hidden dimension, second number of hidden layer before flatten and third is after flatten"}}

        self.D_parser["-nm"] = {"name": "--n_memory", "arg": {"metavar": "n_memory",
                                                              "type": int,
                                                              "default": 5,
                                                              "help": "Number of memory used in giga octet"}}
        self.D_parser["-TU"] = {"name": "--tuning", "arg": {"action": "store_true",
                                                            "help": "Specify to tune or not the model"}}
        self.D_parser["-FT"] = {"name": "--fine_tuning", "arg": {"action": "store_true",
                                                            "help": "Specify to fine tune or not the model"}}
        self.D_parser["-pfsr"] = {"name": "--path_folder_save_read2genome",
                                  "arg": {"metavar": "path_folder_save_read2genome",
                                          "type": str,
                                          "default": None,
                                          "help": "If the path of the folder is given, it will save the matrix returned by read2genome in order to compute it just once"}}
        self.D_parser["-ib"] = {"name": "--is_bok", "arg": {"action": "store_true",
                                                            "help": "If true, specify that the dataframe is formed as BoK."}}
        self.D_parser["-r"] = {"name": "--resources", "arg": {"metavar": "resources",
                                                              "type": str,
                                                              "default": "cpu:1,gpu:0,worker:5",
                                                              "help": "Comma separated list of resources for ray. It is like cpu:x,gpu:y,worker:z where x is the number of cpu per worker, y the number of gpus use per worker (could be fraction) and z the number of workers run in concurrence"}}
        self.D_parser["-nh"] = {"name": "--nhead", "arg": {"metavar": "nhead",
                                                           "type": int,
                                                           "default": 6,
                                                           "help": "Number of head in transformer layers"}}
        self.D_parser["-nl"] = {"name": "--nlayers", "arg": {"metavar": "nlayers",
                                                             "type": int,
                                                             "default": 3,
                                                             "help": "Number of layers in transformer"}}
        self.D_parser["-mv"] = {"name": "--max_vectors", "arg": {"metavar": "max_vectors",
                                                                 "type": int,
                                                                 "default": 100000,
                                                                 "help": "Maximum of vectors used"}}
        self.D_parser["-CL"] = {"name": "--clip", "arg": {"metavar": "clip",
                                                          "type": float,
                                                          "default": 1.,
                                                          "help": "percentage of test data"}}
        self.D_parser["-no"] = {"name": "--noise", "arg": {"metavar": "noise",
                                                           "type": int,
                                                           "default": 0,
                                                           "help": "Mutation rate"}}
        self.D_parser['-pjma'] = {"name": "--path_json_modif_abundance", "arg": {"metavar": "path_json_modif_abundance",
                                                                                 "type": str,
                                                                                 "required": False,
                                                                                 "help": "Path to the json containing tax_id: factor where tax_id is the id at the tax_level taxonomic level and factor is the multiplicator to change the original abundance balanced"}}
        self.D_parser["-go"] = {"name": "--giga_octet", "arg": {"metavar": "giga_octet",
                                                                "type": float,
                                                                "default": 1.,
                                                                "help": "Giga octet simulation by sample"}}
        self.D_parser['-pap'] = {"name": "--path_abundance_profile", "arg": {"metavar": "path_abundance_profile",
                                                                             "type": str,
                                                                             "required": True,
                                                                              "help": "If the abundance profile is a file then this profile is replicated n_sample times, if it is a folder then n_sample is replaced by the total number of abundance profiles."}}
        self.D_parser['-sa'] = {"name": "--simulate_abundance", "arg": {"action": "store_true",
                                                                       "help": "If true create a simulation abundance, else only compute abundance balanced"}}
        self.D_parser['-im'] = {"name": "--in_memory", "arg": {"action": "store_true",
                                                                        "help": "Compute in memory (pandas instead of h2o pysparkling or spark"}}
        self.D_parser["-il"] = {"name": "--id_label", "arg": {"metavar": "id_label",
                                                              "type": str,
                                                              "default": None,
                                                              "help": "Comma separated id like id.fasta,group. If given compute only one element and path_metadata is not required."}}
        self.D_parser["-ftd"] = {"name": "--list_file_to_del", "arg": {"metavar": "list_file_to_del",
                                                                       "type": str,
                                                                       "default": None,
                                                                       "required": False,
                                                                       "help": "Comma separated name of file within path_data that we don't want to process."}}
        self.D_parser["-bi"] = {"name": "--bowtie_index", "arg": {"metavar": "bowtie_index",
                                                                  "type": str,
                                                                  "default": None,
                                                                  "required": False,
                                                                  "help": "Bowtie index used for alignment"}}
        self.D_parser["-pp"] = {"name": "--paired_prediction", "arg": {"action": "store_true",
                                                               "help": "Read that are paired will have the same prediction from read2genome by taking the max probability of both reads."}}
        self.D_parser["-cv"] = {"name": "--cross_validation", "arg": {"metavar": "cross_validation",
                                                              "type": int,
                                                              "default": 20,
                                                              "help": "Number of cross validation"}}
        self.D_parser["-tm"] = {"name": "--to_merge", "arg": {"action": "store_true",
                                                              "help": "Merge output files from CAMISIM because anonymized was set to false."}}
        self.D_parser["-d"] = {"name": "--disease", "arg": {"metavar": "disease",
                                                            "type": str,
                                                            "required": True,
                                                            "help": "Comma separated list of the string corresponding the disease class for each dataset"}}
        self.D_parser["-aa"] = {"name": "--add_abundance", "arg": {"metavar": "add_abundance",
                                                                   "type": str,
                                                                   "default": "no",
                                                                   "choices": ["no", "yes", "only"]}}
        self.D_parser["-AF"] = {"name": "--activation_function", "arg": {"metavar": "activation_function",
                                                                 "type": str,
                                                                 "default": "nn.ReLU",
                                                                 "choices": ["nn.ReLU", "nn.LeakyReLU"],
                                                                 "help": "The activation function used during training."}}

        self.parser_donwload_metagenomic_data()
        self.parser_bok_split()
        self.parser_bok_merge()
        self.parser_clean_raw_metagenomic_data()
        self.parser_metagenome_kmerization()
        self.parser_genome_kmerization()
        self.parser_create_df_fasta_metadata()
        self.parser_create_simulated_read2genome_dataset()
        self.parser_create_simulated_metagenome2vec_dataset()
        self.parser_create_camisim_config_file()
        self.parser_fastdna()
        
    ##############################
    # METAGENOME PROCESSING
    ##############################

    def add_subparser(func_create_parser):
        def wrapper(self, *args, **kwargs):
            D_parser_back = copy.deepcopy(self.D_parser)
            func_create_parser(self, *args, **kwargs)
            self.D_parser = copy.deepcopy(D_parser_back)
        return wrapper

    @add_subparser
    def parser_donwload_metagenomic_data(self):
        parser = self.subparsers.add_parser('download_metagenomic_data')
        for k in ['-pd', '-ps', '-isi', '-iu']:
            if k == '-pd':
                self.D_parser[k]["arg"]["help"] = "Path of the tsv file containing information to download the data."
            parser.add_argument(k, self.D_parser[k]['name'], **self.D_parser[k]['arg'])
    
    @add_subparser
    def parser_bok_split(self):
        parser = self.subparsers.add_parser('bok_split')
        for k in ['-k', '-s', '-lf', '-pg', '-ps', '-np',
                  '-if', '-o', '-mo', '-pd', '-ng']:
            parser.add_argument(k, self.D_parser[k]['name'], **self.D_parser[k]['arg'])

    @add_subparser
    def parser_bok_merge(self):
        parser = self.subparsers.add_parser('bok_merge')
        for k in ['-lf', '-pg', '-pd', '-np', '-mo', '-o', '-ng']:
            parser.add_argument(k, self.D_parser[k]['name'], **self.D_parser[k]['arg'])

    @add_subparser
    def parser_clean_raw_metagenomic_data(self):
        parser = self.subparsers.add_parser('clean_raw_metagenomic_data')
        for k in ['-pd', '-nsl', '-ps', '-if', '-mo', '-np', '-o', '-pg', '-lf', '-ftd', '-im']:
            if k == '-nsl':
                self.D_parser[k]["arg"]["default"] = -1
                self.D_parser[k]["arg"]["help"] = "Maximum number of reads loaded for one sample "
            parser.add_argument(k, self.D_parser[k]['name'], **self.D_parser[k]['arg'])        
    
    @add_subparser
    def parser_metagenome_kmerization(self):
        parser = self.subparsers.add_parser('metagenome_kmerization')
        for k in ['-pd', '-k', '-nsl', '-ps', '-if', '-mo', '-np', '-pg', '-lf', '-im']:
            if k == '-nsl':
                self.D_parser[k]["arg"]["default"] = -1
                self.D_parser[k]["arg"]["help"] = "Maximum number of reads loaded for one sample "
            parser.add_argument(k, self.D_parser[k]['name'], **self.D_parser[k]['arg'])
    
    ##############################
    # GENOME PROCESSING
    ##############################
    
    def parser_genome_kmerization(self):
        parser = self.subparsers.add_parser('genome_kmerization')
        for k in ['-k', '-s', '-pd', '-ps']:
            if k == '-pd':
                self.D_parser[k]["arg"]["help"] = "Path to the genome data"
            if k == '-ps':
                self.D_parser[k]["arg"]["help"] = "Path to the saved kmerized genome"
            parser.add_argument(k, self.D_parser[k]['name'], **self.D_parser[k]['arg'])
    
    ##############################
    # SIMULATION
    ##############################
    
    @add_subparser
    def parser_create_df_fasta_metadata(self):
        parser = self.subparsers.add_parser('create_df_fasta_metadata')
        for k in ['-pd', '-ps', '-pjma', '-pmd', '-sa']:
            if k == '-ps':
                self.D_parser[k]["arg"]["required"] = True
            if k == '-pmd':
                self.D_parser[k]["arg"]["required"] = False
                self.D_parser[k]["arg"]["default"] = None
                self.D_parser[k]["arg"]["help"] = "If given it corresponds to a path of an already existing metadata file, this avoid computation time when generating it"
            if k == '-pd':
                self.D_parser[k]["arg"]["help"] = "Path to the folder containing fasta files."
            parser.add_argument(k, self.D_parser[k]['name'], **self.D_parser[k]['arg'])
    

    @add_subparser
    def parser_create_simulated_read2genome_dataset(self):
        parser = self.subparsers.add_parser('create_simulated_read2genome_dataset')
        for k in ['-pfq', '-pmf', '-pmd', '-ps', '-nsl', '-o', '-vs']:
            if k == '-nsl':
                self.D_parser[k]["arg"]["help"] = "Number of reads taken in the simulation dataset"
                self.D_parser[k]["arg"]["default"] = -1
            if k == '-vs':
                self.D_parser[k]["arg"]["default"] = 0.3
            parser.add_argument(k, self.D_parser[k]['name'], **self.D_parser[k]['arg'])

    @add_subparser
    def parser_create_simulated_metagenome2vec_dataset(self):
        parser = self.subparsers.add_parser('create_simulated_metagenome2vec_dataset')
        for k in ['-pd', '-ps', '-o', '-tm']:
            if k == '-pd':
                self.D_parser[k]["arg"]["help"] = "Comma separated path to the folders containing the simulation data. One folder corresponds to a class of samples."
            parser.add_argument(k, self.D_parser[k]['name'], **self.D_parser[k]['arg'])
    
    @add_subparser
    def parser_create_camisim_config_file(self):
        parser = self.subparsers.add_parser('create_camisim_config_file')
        for k in ['-ps', '-nc', '-nsc', '-ct', '-pt', '-go', '-pap']:
            if k == '-ct':
                self.D_parser[k]["arg"]["choices"] = ["illumina", "nanosim", "both"]
                self.D_parser[k]["arg"]["type"] = str
                self.D_parser[k]["arg"]["default"] = "illumina"
                self.D_parser[k]["arg"]["help"] = "'illumina', 'nanosim' or 'both' simulator"
            if k == '-pd':
                self.D_parser[k]["arg"]["help"] = "Path of the create_df_fasta_metadata output"
            parser.add_argument(k, self.D_parser[k]['name'], **self.D_parser[k]['arg'])

    ##############################
    # READ2GENOME
    ##############################

    @add_subparser
    def parser_fastdna(self):
        parser = self.subparsers.add_parser('fastdna')
        for k in ['-pd', '-k', '-E', '-S', '-R', '-nc', '-prg', '-pkv', '-pt', '-tt', '-no', '-Ml']:
            if k == '-pd':
                self.D_parser[k]["arg"][
                    "help"] = "comma separated path, first is reads fasta file, second is corresponding labels"
                self.D_parser[k]["arg"]["required"] = True
            if k == '-pm':
                self.D_parser[k]["arg"]["help"] = "Complete path to save the model"
                self.D_parser[k]["arg"]["required"] = True
            if k == '-pkv':
                self.D_parser[k]["arg"]["help"] = "Complete path to save final embeddings"
                self.D_parser[k]["arg"]["required"] = False
            parser.add_argument(k, self.D_parser[k]['name'], **self.D_parser[k]['arg'])

    ##############################
    ##############################
    ##############################

    def parser_word2vec(self):
        parser = argparse.ArgumentParser(description='Arguments for the kmer2vec algorithm')
        for k in ['-dn', '-k', '-w', '-lf', '-pg', '-pa', '-pm', '-pl', '-rl', '-nsl', '-nb',
                  '-B', '-E', '-S', '-R', '-V', '-VS', '-VW', '-NS', '-NL', '-NSIM', '-cl']:
            if k == "-pm":
                self.D_parser[k]["arg"]["default"] = "/tmp"
            parser.add_argument(k, self.D_parser[k]['name'], **self.D_parser[k]['arg'])
        return parser.parse_args()

    def parser_word2vec_genome(self):
        parser = argparse.ArgumentParser(description='Arguments for the kmer2vec algorithm')
        for k in ['-pd', '-pa', '-E', '-S', '-R', '-w', '-k', '-s', '-ca', '-sc', '-pg', '-lf', '-nc', '-pt']:
            if k == "-pd":
                self.D_parser[k]["arg"]["help"] = "path to the text file that is the corpus to learn."
            if k == '-pt':
                self.D_parser[k]["arg"]["default"] = './'
            if k == "-R":
                self.D_parser[k]["arg"]["default"] = 0.025
            if k == "-S":
                self.D_parser[k]["arg"]["default"] = 1
            if k == '-s':
                self.D_parser[k]["arg"]["required"] = True
            parser.add_argument(k, self.D_parser[k]['name'], **self.D_parser[k]['arg'])
        return parser.parse_args()


    def parser_GloVe(self):
        parser = argparse.ArgumentParser(description='Arguments for glove learning')
        for k in ['-pl', '-pa', '-vfn', '-cfn', '-mo', '-S', '-E', '-X', '-R',
                  '-k', '-w', '-lf', '-dn', '-pg', '-ps', '-nc']:
            if k == '-R':
                self.D_parser[k]["arg"]["default"] = 0.05
            parser.add_argument(k, self.D_parser[k]['name'], **self.D_parser[k]['arg'])
        return parser.parse_args()

    def parser_GloVe_genome(self):
        parser = argparse.ArgumentParser(description='Arguments for glove learning')
        for k in ['-pd', '-pa', '-vfn', '-cfn', '-S', '-E', '-X', '-R', '-w', '-ps', '-nc', '-k', '-s', '-ca', '-pg',
                  '-lf', '-pt']:
            if k == '-R':
                self.D_parser[k]["arg"]["default"] = 0.05
            if k == '-pt':
                self.D_parser[k]["arg"]["default"] = './'
            if k == '-ps':
                self.D_parser[k]["arg"]["required"] = True
            if k == '-s':
                self.D_parser[k]["arg"]["required"] = True
            if k == "-pd":
                self.D_parser[k]["arg"]["help"] = "path to the text file that is the corpus to learn."
            parser.add_argument(k, self.D_parser[k]['name'], **self.D_parser[k]['arg'])
        return parser.parse_args()

    def parser_fasttext(self):
        parser = argparse.ArgumentParser(description='Arguments for the fasttext algorithm')
        for k in ['-E', '-S', '-R', '-w', '-pg', '-pa', '-dn',
                  '-k', '-lf', '-mng', '-Mng', '-pd', '-nc']:
            if k == "-pd":
                self.D_parser[k]["arg"]["help"] = "path to the text file that is the corpus to learn."
            parser.add_argument(k, self.D_parser[k]['name'], **self.D_parser[k]['arg'])
        return parser.parse_args()

    def parser_fasttext_genome(self):
        parser = argparse.ArgumentParser(description='Arguments for the fasttext algorithm')
        for k in ['-pd', '-pa', '-E', '-S', '-R', '-w', '-mng', '-Mng', '-nc', '-k', '-s', '-ca', '-pg', '-lf', '-pt']:
            if k == '-pt':
                self.D_parser[k]["arg"]["default"] = './'
            if k == "-pd":
                self.D_parser[k]["arg"]["help"] = "path to the text file that is the corpus to learn."
            if k == '-s':
                self.D_parser[k]["arg"]["required"] = True
            parser.add_argument(k, self.D_parser[k]['name'], **self.D_parser[k]['arg'])
        return parser.parse_args()

    def parser_analyse_embeddings(self):
        parser = argparse.ArgumentParser(description='Arguments for the embeddings analysis')
        for k in ['-pd', '-pkv', '-ct', '-nc', '-ni', '-o']:
            if k == '-ct':
                self.D_parser[k]["arg"][
                    "help"] = "Computation type : Comma separated string with each option wanted. 0 t-sne projection, 1 edit vs cosine similarity, 2 needlman-wunsch vs cosine similarity, 3 compute all analysis"
                self.D_parser[k]["arg"]["type"] = str
                del self.D_parser[k]["arg"]["choices"]
            if k == "-ni":
                self.D_parser[k]["arg"]["help"] = "Number of k-mers to compute in the analysis."
                self.D_parser[k]["arg"]["default"] = None
            if k == "-nc":
                self.D_parser[k]["arg"]["default"] = 1
            if k == "-pd":
                self.D_parser[k]["arg"]["required"] = False
                self.D_parser[k]["arg"]["help"] = "Path to the data used for the embeddings training"
            parser.add_argument(k, self.D_parser[k]['name'], **self.D_parser[k]['arg'])
        return parser.parse_args()

    def parser_analyse_matrix_distance(self):
        parser = argparse.ArgumentParser(description='Arguments for the distance matrix analysis')
        for k in ['-pl', '-dn', '-k', '-w', '-ps', '-mo']:
            parser.add_argument(k, self.D_parser[k]['name'], **self.D_parser[k]['arg'])
        return parser.parse_args()

    def parser_deepsets(self):
        parser = argparse.ArgumentParser(description='Arguments for deepsets script')
        for k in ['-pd', '-pmd', '-pm', '-ps', '-dn', '-B', '-S', '-R', '-D', '-TS',
                  '-DO', '-DS', '-ig', '-nm', '-TU', '-I', '-r', '-CL', '-pt', '-d', '-cv']:
            if k == '-dn':
                self.D_parser[k]["arg"]["help"] = "The name of the model to save"
                self.D_parser[k]["arg"]["default"] = "deepsets"
            if k == '-pm':
                self.D_parser[k]["arg"]["help"] = "Path to the folder to save tunning files and model"
                self.D_parser[k]["arg"]["required"] = True
            if k == '-ps':
                self.D_parser[k]["arg"]["help"] = "Complete path to the file where is saved the deepsets scores"
                self.D_parser[k]["arg"]["required"] = True
            if k == '-pd':
                self.D_parser[k]["arg"]["help"] = "Complete path to the MIL matrix file"
                self.D_parser[k]["arg"]["required"] = True
            if k == '-I':
                self.D_parser[k]["arg"]["help"] = "Number of model trained when tunning"
                self.D_parser[k]["arg"]["default"] = 10
            parser.add_argument(k, self.D_parser[k]['name'], **self.D_parser[k]['arg'])
        return parser.parse_args()

    def parser_vae(self):
        parser = argparse.ArgumentParser(description='Arguments for VAE script')
        for k in ['-pd', '-pmd', '-pm', '-dn', '-B', '-S', '-R', '-D', '-d', '-ct',
                  '-DO', '-DV', '-ig', '-nm', '-TU', '-I', '-r', '-CL', '-pt', '-cv', '-TS', '-AF']:
            if k == '-dn':
                self.D_parser[k]["arg"]["help"] = "The name of the model to save"
                self.D_parser[k]["arg"]["default"] = "vae"
            if k == '-ct':
                self.D_parser[k]["arg"]["help"] = "If vae uses variational auto encoder, else if ae uses auto encoder"
                self.D_parser[k]["arg"]["choices"] = ["vae", "ae"]
                self.D_parser[k]["arg"]["default"] = "vae"
                self.D_parser[k]["arg"]["type"] = str
            if k == '-pm':
                self.D_parser[k]["arg"]["help"] = "Path to the folder to save tunning files and model"
                self.D_parser[k]["arg"]["required"] = True
            if k == '-pd':
                self.D_parser[k]["arg"]["help"] = "Complete path to the MIL matrix file"
                self.D_parser[k]["arg"]["required"] = True
            if k == '-I':
                self.D_parser[k]["arg"]["help"] = "Number of model trained when tunning"
                self.D_parser[k]["arg"]["default"] = 10
            parser.add_argument(k, self.D_parser[k]['name'], **self.D_parser[k]['arg'])
        return parser.parse_args()

    def parser_snn(self):
        parser = argparse.ArgumentParser(description='Arguments for SNN script')
        for k in ['-pd', '-pmd', '-pm', '-dn', '-B', '-S', '-R', '-D', '-d',
                  '-DO', '-DV', '-ig', '-nm', '-TU', '-I', '-r', '-CL', '-pt', '-cv', '-TS']:
            if k == '-dn':
                self.D_parser[k]["arg"]["help"] = "The name of the model to save"
                self.D_parser[k]["arg"]["default"] = "vae"
            if k == '-ct':
                self.D_parser[k]["arg"]["help"] = "If vae uses variational auto encoder, else if ae uses auto encoder"
                self.D_parser[k]["arg"]["choices"] = ["vae", "ae"]
                self.D_parser[k]["arg"]["default"] = "vae"
                self.D_parser[k]["arg"]["type"] = str
            if k == '-pm':
                self.D_parser[k]["arg"]["help"] = "Path to the folder to save tunning files and model"
                self.D_parser[k]["arg"]["required"] = True
            if k == '-pd':
                self.D_parser[k]["arg"]["help"] = "Complete path to the MIL matrix file"
                self.D_parser[k]["arg"]["required"] = True
            if k == '-I':
                self.D_parser[k]["arg"]["help"] = "Number of model trained when tunning"
                self.D_parser[k]["arg"]["default"] = 10
            parser.add_argument(k, self.D_parser[k]['name'], **self.D_parser[k]['arg'])
        return parser.parse_args()

    def parser_benchmark(self):
        parser = argparse.ArgumentParser(description='Arguments for benchmark script')
        for k in ['-pd', '-pmd', '-ps', '-dn', '-nc', '-I', '-ct', '-ib', '-pm', '-ig',
                  '-TS', '-cv', '-d', '-TU', '-FT', '-aa']:
            if k == '-pd':
                self.D_parser[k]["arg"]["help"] = "Complete path to the matrix to feed into the benchmark"
                self.D_parser[k]["arg"]["required"] = True
            if k == '-I':
                self.D_parser[k]["arg"]["required"] = True
            if k == '-ps':
                self.D_parser[k]["arg"]["help"] = "Complete path of the file where the results are saved"
                self.D_parser[k]["arg"]["required"] = True
            if k == '-pm':
                self.D_parser[k]["arg"]["help"] = "Complete path to the parameters of the vae model saved in json"
                self.D_parser[k]["arg"]["default"] = None
                self.D_parser[k]["arg"]["required"] = False
            if k == '-ct':
                self.D_parser[k]["arg"]["help"] = "Transformation for compositional data. Could be alr, clr or ilr. If defined, data are reformed as compositional data."
                self.D_parser[k]["arg"]["default"] = None
                self.D_parser[k]["arg"]["required"] = False
                self.D_parser[k]["arg"]["type"] = str
                self.D_parser[k]["arg"]["choices"] = [None, "alr", "clr", "ilr", "vae", "ae", "snn"]
            if k == '-dn':
                self.D_parser[k]["arg"]["help"] = "Name of the dataframe to use"
                self.D_parser[k]["arg"]["required"] = True
            if k == '-fn':
                self.D_parser[k]["arg"]["help"] = "Number of models trained in random grid search."
                self.D_parser[k]["arg"]["required"] = False
                self.D_parser[k]["arg"]["default"] = 100
            parser.add_argument(k, self.D_parser[k]['name'], **self.D_parser[k]['arg'])
        return parser.parse_args()

    def parser_cluster_map(self):
        parser = argparse.ArgumentParser(description='Arguments for cluter map script')
        for k in ['-pd', '-pmd']:
            if k == '-pd':
                self.D_parser[k]["arg"]["help"] = "Complete path to the matrix for the cluster map"
                self.D_parser[k]["arg"]["required"] = True
            if k == '-pmd':
                self.D_parser[k]["arg"][
                    "help"] = "Absolute path to a csv file containing 2 columns : 'id.fasta' (the metagenome's id) and 'group' that is the class of a metagenome"
            parser.add_argument(k, self.D_parser[k]['name'], **self.D_parser[k]['arg'])
        return parser.parse_args()

    def parser_generate_local_kmerized_metagenome(self):
        parser = argparse.ArgumentParser(description='Arguments for ELMo files generation')
        for k in ['-pd', '-pl', '-dn', '-k', '-w', '-ps', '-mo', '-ns', '-pg', '-lf', '-np']:
            parser.add_argument(k, self.D_parser[k]['name'], **self.D_parser[k]['arg'])
        return parser.parse_args()

    def parser_bok(self):
        parser = argparse.ArgumentParser(description="Arguments for structuring metagenomes to BoK representation")
        for k in ['-pd', '-k', '-o', '-pg', '-lf', '-pmd']:
            parser.add_argument(k, self.D_parser[k]['name'], **self.D_parser[k]['arg'])
            if k == '-k':
                self.D_parser[k]["arg"]["required"] = True
            if k == '-pmd':
                self.D_parser[k]["arg"][
                    "help"] = "Absolute path to a csv file containing 2 columns : 'id.fasta' (the metagenome's id) and 'group' that is the class of a metagenome"
            if k == '-pd':
                self.D_parser[k]["arg"]["required"] = True
                self.D_parser[k]["arg"][
                    "help"] = "Path were are saved the BoK files for each metagenomes. It is also where is saved the BoK final matrix with metagenome info concatenated"
        return parser.parse_args()

    def parser_metagenome2vec(self):
        parser = argparse.ArgumentParser(description="Arguments for transformation metagenomes' reads to embeddings")
        for k in ['-ps', '-k', '-np', '-mo', '-pd', '-lf', '-T', '-prv', '-pt', '-o', '-pg', '-ng', '-im',
                  '-rv', '-ig', '-pmwc', '-ct', '-pmd', '-nsl', '-prg', '-ni', '-rg', '-pfsr', '-pmca', '-il', '-pp']:
            if k == '-f':
                self.D_parser[k]["arg"]["required"] = True
                self.D_parser[k]["arg"]["help"] = "The suffix name of the matrix file saved. To identify all matrix"
            if k == '-m':
                self.D_parser[k]["arg"]["default"] = "mean"
                self.D_parser[k]["arg"]["help"] = "Computation method : mean or sum are available"
            if k == '-ct':
                self.D_parser[k]["arg"][
                    "help"] = "Comma separated string with value: 0 tabular, 1 MIL, 2 for cut metagenome analysis"
                self.D_parser[k]["arg"]["type"] = str
                del self.D_parser[k]["arg"]["choices"]
            if k == '-B':
                self.D_parser[k]["arg"]["default"] = 1e6
            if k == '-pal':
                self.D_parser[k]["arg"]["required"] = False
            if k == '-k':
                self.D_parser[k]["arg"]["required"] = False
            if k == '-rg':
                self.D_parser[k]["arg"]["required"] = False
            if k == '-pm':
                self.D_parser[k]["arg"]["required"] = False
                self.D_parser[k]["arg"]["default"] = None
            if k == '-ca':
                self.D_parser[k]["arg"]["required"] = False
            if k == '-dn':
                self.D_parser[k]["arg"]["required"] = True
            if k == '-nsl':
                self.D_parser[k]["arg"]["default"] = -1
            if k == '-pmd':
                self.D_parser[k]["arg"][
                    "help"] = "Absolute path to a csv file containing 2 columns : 'id.fasta' (the metagenome's id) and 'group' that is the class of a metagenome"
                self.D_parser[k]["arg"]["required"] = False
                self.D_parser[k]["arg"]["default"] = None
            if k == '-m':
                self.D_parser[k]["arg"]["default"] = None
                self.D_parser[k]["arg"]["help"] = "The path where is saved the read2genome model"
            if k == '-T':
                self.D_parser[k]["arg"]["default"] = 0.0
                self.D_parser[k]["arg"]["type"] = float
                self.D_parser[k]["arg"]["help"] = "Value of the threshold for the read2genome"
            parser.add_argument(k, self.D_parser[k]['name'], **self.D_parser[k]['arg'])
        return parser.parse_args()

    def parser_seq2seq(self):
        parser = argparse.ArgumentParser(description="Arguments for transformation metagenomes' reads to embeddings")
        for k in ['-k', '-w', '-pa', '-pal', '-s', '-kea', '-dn', '-ca', '-S', '-B', '-ig', '-R', '-E', '-pd',
                  '-I', '-nco', '-Ml', "-nc", '-lf', '-pg', '-lm']:
            if k == '-pd':
                self.D_parser[k]["arg"]["required"] = True
                self.D_parser[k]["arg"]["help"] = "Comma separated path first for training second for validation"
            if k == '-f':
                self.D_parser[k]["arg"]["required"] = True
                self.D_parser[k]["arg"]["help"] = "The file name to save the pytorch model learnt"
            if k == '-ca':
                self.D_parser[k]["arg"]["required"] = False
            if k == '-S':
                self.D_parser[k]["arg"]["default"] = 10
            if k == "-w":
                self.D_parser[k]["arg"]["required"] = False
                self.D_parser[k]["arg"]["default"] = None
            if k == "-k":
                self.D_parser[k]["arg"]["required"] = False
            if k == "-pal":
                self.D_parser[k]["arg"]["required"] = False
                self.D_parser[k]["arg"]["default"] = None
            if k == "-kea":
                self.D_parser[k]["arg"]["default"] = None
            parser.add_argument(k, self.D_parser[k]['name'], **self.D_parser[k]['arg'])
        return parser.parse_args()

    def parser_transformer(self):
        parser = argparse.ArgumentParser(description="Arguments for transofmers model")
        for k in ['-k', '-f', '-S', '-I', '-B', '-ig', '-nc', '-R', '-E', '-nco', '-DO', '-mv',
                  '-pkv', '-pa', '-pd', '-Ml', '-pg', '-lf', '-H', '-lm', '-nh', '-nl', '-pkc', '-o', '-CL']:
            if k == '-pd':
                self.D_parser[k]["arg"]["required"] = True
                self.D_parser[k]["arg"]["help"] = "Comma separated path first for training second for validation"
            if k == '-f':
                self.D_parser[k]["arg"]["required"] = True
                self.D_parser[k]["arg"]["help"] = "The file name to save the pytorch model learnt"
            if k == '-S':
                self.D_parser[k]["arg"]["default"] = 10
            if k == '-E':
                self.D_parser[k]["arg"]["default"] = 200
            if k == '-H':
                self.D_parser[k]["arg"]["default"] = 200
            if k == '-DO':
                self.D_parser[k]["arg"]["default"] = 0.2
            if k == '-DO':
                self.D_parser[k]["arg"]["default"] = 0.001
            if k == "-k":
                self.D_parser[k]["arg"]["required"] = False
            if k == "-pal":
                self.D_parser[k]["arg"]["required"] = False
                self.D_parser[k]["arg"]["default"] = None
            if k == "-kea":
                self.D_parser[k]["arg"]["default"] = None
            parser.add_argument(k, self.D_parser[k]['name'], **self.D_parser[k]['arg'])
        return parser.parse_args()

    def parser_transformer_classifier(self):
        parser = argparse.ArgumentParser(description="Arguments for transofmers model")
        for k in ['-k', '-f', '-S', '-I', '-B', '-ig', '-nc', '-R', '-E', '-DO',
                  '-pkv', '-pa', '-pd', '-Ml', '-pg', '-lf', '-H', '-nh', '-nl', "-tl"]:
            if k == '-pd':
                self.D_parser[k]["arg"]["required"] = True
                self.D_parser[k]["arg"]["help"] = "Comma separated path first for training second for validation"
            if k == '-f':
                self.D_parser[k]["arg"]["required"] = True
                self.D_parser[k]["arg"]["help"] = "The file name to save the pytorch model learnt"
            if k == '-S':
                self.D_parser[k]["arg"]["default"] = 10
            if k == '-E':
                self.D_parser[k]["arg"]["default"] = 200
            if k == '-H':
                self.D_parser[k]["arg"]["default"] = 200
            if k == '-DO':
                self.D_parser[k]["arg"]["default"] = 0.2
            if k == '-DO':
                self.D_parser[k]["arg"]["default"] = 0.001
            if k == "-k":
                self.D_parser[k]["arg"]["required"] = False
            if k == "-pal":
                self.D_parser[k]["arg"]["required"] = False
                self.D_parser[k]["arg"]["default"] = None
            if k == "-kea":
                self.D_parser[k]["arg"]["default"] = None
            parser.add_argument(k, self.D_parser[k]['name'], **self.D_parser[k]['arg'])
        return parser.parse_args()

    def parser_train_h2o_model(self):
        parser = argparse.ArgumentParser(description="Arguments for transformation metagenomes' reads to embeddings")
        for k in ['-k', '-rv', '-pd', '-o', '-pm', '-f', "-mla", "-mm",
                  '-pmwc', '-pg', '-lf', '-ig', '-mo', '-np', '-tl', '-nsl', '-ps', '-tt', '-nf',
                  '-prv', '-pt', '-ot', '-pmd']:
            if k == '-f':
                self.D_parser[k]["arg"]["help"] = "Name of the read2genome model when saved"
                self.D_parser[k]["arg"]["required"] = True
            if k == '-k':
                self.D_parser[k]["arg"]["required"] = False
            if k == '-pd':
                self.D_parser[k]["arg"]["required"] = False
                self.D_parser[k]["arg"][
                    "help"] = "Two complete path separated by a coma respectively for train and valid data set"
                self.D_parser[k]["arg"]["default"] = 10
            if k == '-np':
                self.D_parser[k]["arg"]["default"] = None
            if k == '-nsl':
                self.D_parser[k]["arg"]["default"] = -1
            if k == "-ps":
                self.D_parser[k]["arg"]["required"] = True
                self.D_parser[k]["arg"][
                    "help"] = "Path were is save the parquet file after reads transformation of simulation data"
            parser.add_argument(k, self.D_parser[k]['name'], **self.D_parser[k]['arg'])
        return parser.parse_args()

    def parser_analyse_read2genome(self):
        parser = argparse.ArgumentParser(description="Arguments for analyse read2genome")
        for k in ['-pm', '-pd', '-pg', '-lf', '-mo', '-tl', '-nsl', '-nc', '-rg', '-pt', '-tt', '-ni', '-pmd', '-bi']:
            if k == '-pd':
                self.D_parser[k]["arg"]["required"] = False
                self.D_parser[k]["arg"][
                    "help"] = "Two complete path separated by a coma respectively for train and valid data set"
            if k == '-pm':
                self.D_parser[k]["arg"]["help"] = "Path where is saved the read2genome model"
            if k == '-nsl':
                self.D_parser[k]["arg"]["default"] = -1
            if k == '-ni':
                self.D_parser[k]["arg"]["default"] = 10000
                self.D_parser[k]["arg"]["help"] = "Number of reads computed for bowtie score"
            parser.add_argument(k, self.D_parser[k]['name'], **self.D_parser[k]['arg'])
        return parser.parse_args()

    def parser_genome_projection(self):
        parser = argparse.ArgumentParser(description="Arguments for genomes projection")
        for k in ['-k', '-pd', '-ps', '-o', '-prv', '-rv', '-nc', '-pmwc', '-ig',
                  '-pt', '-pmd', '-lf', '-pg', '-Ml', '-pgd', '-np']:
            if k == '-pd':
                self.D_parser[k]["arg"]["help"] = "Path to the genome data folder containing the fasta files"
                self.D_parser[k]["arg"]["required"] = True
            if k == '-pmd':
                self.D_parser[k]["arg"][
                    "help"] = "Absolute path to the metadata file composed by 4 columns tax_id, genus, family, fasta"
                self.D_parser[k]["arg"]["required"] = True
            if k == '-ps':
                self.D_parser[k]["arg"]["help"] = "Path to the saved kmerized genome"
                self.D_parser[k]["arg"]["required"] = True
            parser.add_argument(k, self.D_parser[k]['name'], **self.D_parser[k]['arg'])
        return parser.parse_args()

    def parser_kmer_count(self):
        parser = argparse.ArgumentParser(description="Arguments for kmer count")
        for k in ['-pd', '-ps', '-mo']:
            if k == '-pd':
                self.D_parser[k]["arg"]["help"] = "Path to the kmerized genome file"
                self.D_parser[k]["arg"]["required"] = True
            if k == '-ps':
                self.D_parser[k]["arg"]["help"] = "Path to the results"
                self.D_parser[k]["arg"]["required"] = True
            if k == '-mo':
                self.D_parser[k]["arg"][
                    "help"] = "'word_count_matrix' if the file given is the output from word_count_split and word_count_merge or 'kmerized_genome_dataset' if the file is the output of kmerization"
                self.D_parser[k]["arg"]["default"] = "word_count_matrix"
                self.D_parser[k]["arg"]["choices"] = ["word_count_matrix", "kmerized_genome_dataset"]
            parser.add_argument(k, self.D_parser[k]['name'], **self.D_parser[k]['arg'])
        return parser.parse_args()
    
    def parser_download_metagenomic_data(self):
        parser = self.subparsers.add_parser('download_metagenomic_data')
        for k in ['-pd', '-ps']:
            parser.add_argument(k, self.D_parser[k]['name'], **self.D_parser[k]['arg'])
        


if __name__ == "__main__":

    parserCreator = ParserCreator()
    args = parserCreator.parser.parse_args()
    # TODO replace with os.path + increment log file if already exists add path log
    dict_commands_metadata = {"download_metagenomic_data": {"path_log": "download", "log_file": "download_metagenomic_data.log", "need_spark": False},
                            "clean_raw_metagenomic_data": {"path_log": "metagenome_preprocessing", "log_file": "clean_raw_metagenomic_data.log", "need_spark": True},
                            "bok_split": {"path_log": "metagenome_preprocessing", "log_file": "bok_split.log", "need_spark": True},
                            "bok_merge": {"path_log": "metagenome_preprocessing", "log_file": "bok_merge.log", "need_spark": True},
                            "create_camisim_config_file": {"path_log": "simulation", "log_file": "create_camisim_config_file.log", "need_spark": False},
                            "run_camisim": {"path_log": "simulation", "log_file": "run_camisim.log", "need_spark": False},
                            "create_df_fasta_metadata": {"path_log": "simulation", "log_file": "create_df_fasta_metadata.log", "need_spark": False},
                            "create_simulated_read2genome_dataset": {"path_log": "simulation", "log_file": "create_simulated_read2genome_dataset.log", "need_spark": False},
                            "create_simulated_metagenome2vec_dataset": {"path_log": "simulation", "log_file": "create_simulated_metagenome2vec_dataset.log", "need_spark": False},
                            "fastdna": {"path_log": "read2genome", "log_file": "fastdna.log", "need_spark": False}}
    
    command_metadata = dict_commands_metadata[args.command]
    path_log, log_file, need_spark = os.path.join(SCRIPT_DIR, "logs", command_metadata["path_log"]), command_metadata["log_file"], command_metadata["need_spark"]
    os.makedirs(path_log, exist_ok=True)
    logging.basicConfig(filename=os.path.join(path_log, log_file), level=logging.INFO)

    spark = None
    if need_spark:
        spark = spark_manager.createSparkSession(args.command)

    if args.command == "download_metagenomic_data":
        download_from_tsv_file(args.path_data, args.path_save, args.index_sample_id, args.index_url)

    if args.command == "clean_raw_metagenomic_data":
        os.makedirs(args.path_save, exist_ok=True)
        logging.info("Start clean_raw_metagenomic_data")
        if not args.is_file:
            for folder in os.listdir(args.path_data):
                logging.info("cleaning metagenome {}".format(os.path.join(args.path_data, folder)))
                preprocess_metagenomic_data(spark, os.path.join(args.path_data, folder), args.path_save, args.n_sample_load, args.num_partitions, args.mode, args.in_memory, args.overwrite)
        else:
            preprocess_metagenomic_data(spark, args.path_data, args.path_save, args.n_sample_load, args.num_partitions, args.mode, args.in_memory, args.overwrite)

    if args.command == "bok_split":
        os.makedirs(args.path_save, exist_ok=True)
        logging.info("Start bok_split")
        if not args.is_file:
            for folder in os.listdir(args.path_data):
                logging.info("Computing bok for metagenome metagenome {}".format(os.path.join(args.path_data, folder)))
                bok_split(spark, os.path.join(args.path_data, folder), args.path_save, args.k_mer_size, args.step, args.mode, args.num_partitions, args.overwrite)
        else:
             bok_split(spark, args.path_data, args.path_save, args.k_mer_size, args.step, args.mode, args.num_partitions, args.overwrite)
   
    if args.command == "bok_merge":
        logging.info("Start bok_merge")
        bok_merge(spark, args.path_data, args.nb_metagenome, args.num_partitions, args.mode, args.overwrite)


    if args.command == "create_camisim_config_file":
        create_simulated_config_file(args.n_cpus, args.n_sample_by_class, args.computation_type, args.giga_octet,
                                args.path_tmp_folder, args.path_save, args.path_abundance_profile)

    if args.command == "create_df_fasta_metadata":
        create_df_fasta_metadata(args.path_data, args.path_save, args.path_json_modif_abundance, args.path_metadata, args.simulate_abundance)
    
    if args.command == "create_simulated_read2genome_dataset":
        create_simulated_read2genome_dataset(args.path_fastq_file, args.path_mapping_file, args.path_metadata, args.path_save,
                                            args.valid_size, args.n_sample_load, args.overwrite)
    if args.command == "create_simulated_metagenome2vec_dataset":
        create_simulated_metagenome2vec_dataset(args.path_data, args.path_save, args.overwrite, args.to_merge)

    if args.command == "fastdna":
        fastdna = FastDnaPred()
        fastdna.train(args.path_data, args.k_mer_size, args.embedding_size, args.n_steps, args.learning_rate, args.tax_taken,
                    args.path_kmer2vec, args.path_read2genome, args.path_tmp_folder, args.n_cpus, args.noise, args.max_length)
    logging.info("End computing")



"""
if __name__ == "__main__":
    logging.getLogger('pyspark').setLevel(logging.ERROR)
    logging.getLogger("py4j").setLevel(logging.ERROR)

    spark = spark_manager.createSparkSession("clean_raw_data")

    preprocess_metagenomic_data(args.path_data, args.path_save, spark, args.n_sample_load, args.mode,
                                overwrite=args.overwrite, num_partitions=args.num_partitions, in_memory=args.in_memory)

if __name__ == "__main__":
    args = parser_creator.ParserCreator().parser_bok_merge()

    logging.basicConfig(filename=os.path.join(args.path_log, args.log_file), level=logging.DEBUG)

    spark = spark_manager.createSparkSession("BoK merge")

    logging.info("Start computing")
    bok_merge(spark, args.path_data, args.nb_metagenome, args.num_partitions, args.mode, args.overwrite)
    logging.info("End computing")


if __name__ == "__main__":
    args = parser_creator.ParserCreator().parser_metagenomic_kmerization()

    logging.basicConfig(filename=os.path.join(args.path_log, args.log_file), level=logging.DEBUG)

    spark = spark_manager.createSparkSession("Metagenome kmerization")

    kmerize_metagenomic_data(spark, args.path_data, args.path_save, args.k_mer_size, args.n_sample_load,
                             num_partitions=args.num_partitions, in_memory=args.in_memory)
"""