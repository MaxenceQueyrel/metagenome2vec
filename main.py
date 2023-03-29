import os
import argparse
import json
import yaml
import logging
import copy
from metagenome2vec.utils import data_manager, spark_manager, string_names
from metagenome2vec.data_processing.metagenome import preprocess_metagenomic_data, bok_split, bok_merge
from metagenome2vec.data_processing.download_metagenomic_data import download_from_tsv_file
from metagenome2vec.data_processing.simulation import *
from metagenome2vec.read2genome.fastDnaPred import FastDnaPred
from metagenome2vec.read2vec.fastDnaEmbed import FastDnaEmbed
from metagenome2vec.metagenome2vec import bok, embedding
from metagenome2vec.NN import utils as nn_utils
from metagenome2vec.NN.data import load_several_matrix_for_learning

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

class ParserCreator(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser('Main command')
        self.subparsers = self.parser.add_subparsers(help="test", dest="command")
        self.D_parser = {}
        self.D_parser["-k"] = {"name": "--k-mer-size", "arg": {"metavar": "k_mer_size",
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
        self.D_parser["-lf"] = {"name": "--log-file", "arg": {"metavar": "log_file",
                                                              "type": str,
                                                              "default": "log_file.txt",
                                                              "help": "Name of the log file"}}
        self.D_parser["-pg"] = {"name": "--path-log", "arg": {"metavar": "path_log",
                                                              "type": str,
                                                              "default": "./",
                                                              "help": "local path where is stored the log file"}}
        self.D_parser["-pl"] = {"name": "--path-learning", "arg": {"metavar": "path_learning",
                                                                   "type": str,
                                                                   "default": "hdfs://ma-1-1-t630.infiniband:8020/user/mqueyrel/deepGene/data/learning/",
                                                                   "help": "Path where are saved the structuring matrix."}}
        self.D_parser["-pa"] = {"name": "--path-analysis", "arg": {"metavar": "path_analyses",
                                                                   "type": str,
                                                                   "default": "/data/projects/deepgene/analyses/",
                                                                   "help": "Path where are saved the data to analyse."}}
        self.D_parser["-pm"] = {"name": "--path-model", "arg": {"metavar": "path_model",
                                                                "type": str,
                                                                "default": "./",
                                                                "help": "Path where are stored the trained machine learning models"}}
        self.D_parser["-pd"] = {"name": "--path-data", "arg": {"metavar": "path_data",
                                                               "type": str,
                                                               "required": True,
                                                               "help": "Path to the data."}}
        self.D_parser["-ps"] = {"name": "--path-save", "arg": {"metavar": "path_save",
                                                               "type": str,
                                                               "required": True,
                                                               "help": "Path where will be saved the data."}}
        self.D_parser["-pkc"] = {"name": "--path-kmer-count", "arg": {"metavar": "path_kmer_count",
                                                                      "type": str,
                                                                      "default": None,
                                                                      "help": "Path to the kmer count"}}
        self.D_parser["-pmd"] = {"name": "--path-metadata", "arg": {"metavar": "path_metadata",
                                                                    "type": str,
                                                                    "required": True,
                                                                    "help": "Absolute path to the metadata file"}}
        self.D_parser["-pfq"] = {"name": "--path-fastq-file", "arg": {"metavar": "path_fastq_file",
                                                                      "type": str,
                                                                      "required": True,
                                                                      "help": "Absolute path to the fastq file from the simulation"}}
        self.D_parser["-pmf"] = {"name": "--path-mapping-file", "arg": {"metavar": "path_mapping_file",
                                                            "type": str,
                                                            "required": True,
                                                            "help": "Absolute path to the read mapping file from the simulation"}}
        self.D_parser["-f"] = {"name": "--f-name", "arg": {"metavar": "f_name",
                                                           "type": str,
                                                           "default": None,
                                                           "help": "Full path name of the file to structure, if None the whole dataset is stuctured"}}
        self.D_parser["-fl"] = {"name": "--list-file", "arg": {"metavar": "list_file",
                                                               "type": str,
                                                               "default": None,
                                                               "help": "Comma separated string with file name"}}
        self.D_parser["-dn"] = {"name": "--dataset-name", "arg": {"metavar": "dataset_name",
                                                                  "type": str,
                                                                  "default": "dataset",
                                                                  "help": "If f_name is none this is the name given to the files computed and saved"}}
        self.D_parser["-V"] = {"name": "--vocabulary-size", "arg": {"metavar": "vocabulary_size",
                                                                    "type": int,
                                                                    "default": 4 ** 6,
                                                                    "help": "Number of words/chains considered (more frequent)"}}
        self.D_parser["-ct"] = {"name": "--computation-type", "arg": {"metavar": "computation_type",
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
        self.D_parser["-pas"] = {"name": "--parameter-structu", "arg": {"metavar": "parameter_structu",
                                                                        "type": str,
                                                                        "required": True,
                                                                        "help": "The folder containing the file structured with these parameters"}}
        self.D_parser["-pal"] = {"name": "--parameter-learning", "arg": {"metavar": "parameter_learning",
                                                                         "type": str,
                                                                         "required": True,
                                                                         "help": "The parameters used during the embeddings learning."}}
        self.D_parser["-rl"] = {"name": "--ratio-learning", "arg": {"metavar": "ratio_learning",
                                                                    "type": float,
                                                                    "default": 1.,
                                                                    "help": "Take rl percent of the dataframe to run kmer2vec"}}
        self.D_parser["-nsl"] = {"name": "--n-sample-load", "arg": {"metavar": "n_sample_load",
                                                                    "type": int,
                                                                    "default": 1e7,
                                                                    "help": "Number of sampled load in memory, if -1 load all the samples."}}
        self.D_parser["-B"] = {"name": "--batch-size", "arg": {"metavar": "batch_size",
                                                               "type": int,
                                                               "default": 64,
                                                               "help": "Number of elements passed to the learning"}}
        self.D_parser["-E"] = {"name": "--embedding-size", "arg": {"metavar": "embeddings_size",
                                                                   "type": int,
                                                                   "default": 300,
                                                                   "help": "Dimension of the embedding vector"}}
        self.D_parser["-H"] = {"name": "--hidden-size", "arg": {"metavar": "hidden_size",
                                                                "type": int,
                                                                "default": 200,
                                                                "help": "Hidden dimension of the neural network"}}
        self.D_parser["-S"] = {"name": "--n-steps", "arg": {"metavar": "n_steps",
                                                            "type": int,
                                                            "default": 50001,
                                                            "help": "Number of steps during learning"}}
        self.D_parser["-I"] = {"name": "--n-iterations", "arg": {"metavar": "n_iterations",
                                                                 "type": int,
                                                                 "default": -1,
                                                                 "help": "Number of iterations in one step during learning."}}
        self.D_parser["-R"] = {"name": "--learning-rate", "arg": {"metavar": "learning_rate",
                                                                  "type": float,
                                                                  "default": 1.,
                                                                  "help": "Learning rate for the gradient descent"}}
        self.D_parser["-VS"] = {"name": "--valid-size", "arg": {"metavar": "valid_size",
                                                                "type": float,
                                                                "default": 8,
                                                                "help": "Random set for evaluation"}}
        self.D_parser["-vs"] = {"name": "--valid-size", "arg": {"metavar": "valid_size",
                                                                "type": float,
                                                                "default": None,
                                                                "help": "The percentage amount of data in the validation set"}}
        self.D_parser["-VW"] = {"name": "--valid-window", "arg": {"metavar": "valid_window",
                                                                  "type": int,
                                                                  "default": 100,
                                                                  "help": "(kmer2vec) Select VS examples for the validation in the top VW of the distribution"}}
        self.D_parser["-NS"] = {"name": "--num-sampled", "arg": {"metavar": "num_sampled",
                                                                 "type": int,
                                                                 "default": 8,
                                                                 "help": "(kmer2vec) Number of negative examples to sample"}}
        self.D_parser["-NL"] = {"name": "--n-compute-loss", "arg": {"metavar": "n_compute_loss",
                                                                    "type": int,
                                                                    "default": 10,
                                                                    "help": "Number of time you want to calculate the loss"}}
        self.D_parser["-NSIM"] = {"name": "--n-show-similarity", "arg": {"metavar": "n_show_similarity",
                                                                         "type": int,
                                                                         "default": 3,
                                                                         "help": "(kmer2vec) Number of time you want to calculate the similarity in the neural network"}}
        self.D_parser["-cl"] = {"name": "--continue-learning", "arg": {"action": "store_true",
                                                                       "help": "If True restore the previous graph / session and continue the learning from this point"}}
        self.D_parser["-m"] = {"name": "--method", "arg": {"metavar": "method",
                                                           "type": str,
                                                           "default": "normal",
                                                           "help": "The method used to perform"}}
        self.D_parser["-kea"] = {"name": "--kmer-embeddings_algorithm", "arg": {"metavar": "kmer_embeddings_algorithm",
                                                                                "type": str,
                                                                                "default": "fasttext",
                                                                                "help": "Name of the algorithm used for the embedding"}}
        self.D_parser["-if"] = {"name": "--is-file", "arg": {"action": "store_true",
                                                             "help": "If true, sample only a file else a folder."}}
        self.D_parser["-vfn"] = {"name": "--vocabulary-file-name", "arg": {"metavar": "vocabulary_file_name",
                                                                           "type": str,
                                                                           "default": "vocab.txt",
                                                                           "help": "(glove) This is the name of the word count / vocabulary file."}}
        self.D_parser["-cfn"] = {"name": "--cooccurrence-file-name", "arg": {"metavar": "cooccurrence_file_name",
                                                                             "type": str,
                                                                             "default": "cooccurrence.bin",
                                                                             "help": "(glove) This is the name of the cooccurrence file."}}
        self.D_parser["-cfn"] = {"name": "--cooccurrence-file-name", "arg": {"metavar": "cooccurrence_file_name",
                                                                             "type": str,
                                                                             "default": "cooccurrence.bin",
                                                                             "help": "(glove) This is the name of the cooccurrence file."}}
        self.D_parser["-iuk"] = {"name": "--include-unk-kmer", "arg": {"action": "store_true",
                                                                       "help": "(structuring classif) If true, make structuring with unk kmer else avoid them"}}
        self.D_parser["-fn"] = {"name": "--file-name", "arg": {"metavar": "file_name",
                                                               "type": str,
                                                               "required": True,
                                                               "help": "The name given to the res file"}}
        self.D_parser["-X"] = {"name": "--x-max", "arg": {"metavar": "x_max",
                                                          "type": int,
                                                          "default": 10,
                                                          "help": "(glove) Threashold for extremely common word pairs"}}
        self.D_parser["-lm"] = {"name": "--language-modeling", "arg": {"action": "store_true",
                                                                       "help": "Tells  if the trasnformer train like a language modeling model or not"}}
        self.D_parser["-mng"] = {"name": "--min-ngram", "arg": {"metavar": "min_ngram",
                                                                "type": int,
                                                                "default": 3,
                                                                "help": "Minimum size of ngram"}}
        self.D_parser["-Mng"] = {"name": "--max-ngram", "arg": {"metavar": "max_ngram",
                                                                "type": int,
                                                                "default": 6,
                                                                "help": "Maximuml size of ngram"}}
        self.D_parser["-ng"] = {"name": "--nb-metagenome", "arg": {"metavar": "nb_metagenome",
                                                                   "type": int,
                                                                   "default": 10000,
                                                                   "help": "Number of metagenome"}}
        self.D_parser["-nsc"] = {"name": "--n-sample-by-class", "arg": {"metavar": "n_sample_by_class",
                                                                        "type": int,
                                                                        "default": 10000,
                                                                        "help": "Number of samples by class when structuring simulation data."}}
        self.D_parser["-nc"] = {"name": "--n-cpus", "arg": {"metavar": "n_cpus",
                                                            "type": int,
                                                            "default": 16,
                                                            "help": "Number of process used"}}
        self.D_parser["-itf"] = {"name": "--index-tmp-file", "arg": {"metavar": "index_tmp_file",
                                                                     "type": int,
                                                                     "default": -1,
                                                                     "help": "if stm = 2 then it will start from the index given for the next temporary matrix"}}
        self.D_parser["-isi"] = {"name": "--index-sample-id", "arg": {"metavar": "index_sample_id",
                                                                     "type": int,
                                                                     "default": 1,
                                                                     "help": "Index in the tsv file of the column containing the sample ids."}}
        self.D_parser["-iu"] = {"name": "--index-url", "arg": {"metavar": "index_url",
                                                                "type": int,
                                                                "default": 10,
                                                                "help": "Index in the tsv file of the column containing the sample url."}}
        self.D_parser["-np"] = {"name": "--num-partitions", "arg": {"metavar": "num_partitions",
                                                                    "type": int,
                                                                    "default": 16,
                                                                    "help": "Number of partitions of the rdd file"}}
        self.D_parser["-ns"] = {"name": "--nb-sequences-by-metagenome", "arg": {"metavar": "nb_sequences_by_metagenome",
                                                                                "type": int,
                                                                                "required": True,
                                                                                "help": "Number of sequences by metagenome"}}
        self.D_parser["-MIL"] = {"name": "--multi-instance-layer", "arg": {"metavar": "multi_instance_layer",
                                                                           "type": str,
                                                                           "choices": ["sum", "max", "attention"],
                                                                           "default": "sum",
                                                                           "help": "Define the type of mil layer"}}
        self.D_parser["-TS"] = {"name": "--test-size", "arg": {"metavar": "test_size",
                                                               "type": float,
                                                               "default": 0.2,
                                                               "help": "percentage of test data"}}
        self.D_parser["-ig"] = {"name": "--id-gpu", "arg": {"metavar": "id_gpu",
                                                            "type": str,
                                                            "default": "-1",
                                                            "help": "Comma separated string: Index of the gpus we want to use. If -1 use cpu"}}
        self.D_parser["-nb"] = {"name": "--n-batch", "arg": {"metavar": "n_batch",
                                                             "type": int,
                                                             "default": 1e5,
                                                             "help": "Number of batchs generated to gain in computation time. If too big can raise an OOM"}}
        self.D_parser["-ni"] = {"name": "--n-instance", "arg": {"metavar": "n_instance",
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
        self.D_parser["-nri"] = {"name": "--n-reads-instance", "arg": {"metavar": "n_reads_instance",
                                                                       "type": int,
                                                                       "default": 1e5,
                                                                       "help": "Number of reads in one instance of a bag for read embeddings computation."}}
        self.D_parser["-D"] = {"name": "--weight-decay", "arg": {"metavar": "weight_decay",
                                                                 "type": float,
                                                                 "default": 1e-5,
                                                                 "help": "Decay for L2 normalization"}}
        self.D_parser["-ca"] = {"name": "--catalog", "arg": {"metavar": "catalog",
                                                             "type": str,
                                                             "required": True,
                                                             "help": "Name of the catalog used"}}
        self.D_parser["-owc"] = {"name": "--only-word-count", "arg": {"action": "store_true",
                                                                      "help": "Compute only the word count matrix"}}
        self.D_parser["-prt"] = {"name": "--path-read-transformed", "arg": {"metavar": "path_read_transformed",
                                                                            "type": str,
                                                                            "required": True,
                                                                            "help": "Path where are saved the metagenomes with reads transform into embeddings"}}
        self.D_parser["-nco"] = {"name": "--nb-cutoffs", "arg": {"metavar": "nb_cutoffs",
                                                                 "type": int,
                                                                 "default": -1,
                                                                 "help": "Number of cut off for the adaptive softmax"}}
        self.D_parser["-pmwc"] = {"name": "--path-metagenome-word-count",
                                  "arg": {"metavar": "path_metagenome_word_count",
                                          "type": str,
                                          "default": None,
                                          "help": "Complet path to the metagenome word count"}}
        self.D_parser["-pgd"] = {"name": "--path-genome-dist", "arg": {"metavar": "path_genome_dist",
                                                                       "type": str,
                                                                       "default": None,
                                                                       "help": "Path to the genome distance matrix computed by https://gitlab.pasteur.fr/GIPhy/JolyTree/-/blob/master/README.md"}}
        self.D_parser["-Ml"] = {"name": "--max-length", "arg": {"metavar": "max_length",
                                                                "type": int,
                                                                "default": 20,
                                                                "help": "Maximum size of read proceeded by the algorithm"}}
        self.D_parser["-sc"] = {"name": "--spark-conf", "arg": {"metavar": "spark_conf",
                                                                "type": json.loads,
                                                                "default": {},
                                                                "help": "Dict to set the spark conf (avoid to use spark-submit)"}}
        self.D_parser["-tl"] = {"name": "--tax-level", "arg": {"metavar": "tax_level",
                                                               "type": str,
                                                               "choices": ["tax_id", "species", "genus", "family"],
                                                               "default": "species",
                                                               "help": "Determine the taxonomy level considered"}}
        self.D_parser["-T"] = {"name": "--threshold", "arg": {"metavar": "threshold",
                                                               "type": str,
                                                               "default": "0.5",
                                                               "help": "Threshold to accept the prediction."}}
        self.D_parser["-ba"] = {"name": "--balance", "arg": {"metavar": "balance",
                                                             "type": str,
                                                             "choices": [None, "under", "over", "both"],
                                                             "default": None,
                                                             "help": "Tell the method to balance the data"}}
        self.D_parser["-tt"] = {"name": "--tax-taken", "arg": {"metavar": "tax_taken",
                                                               "type": str,
                                                               "default": None,
                                                               "help": "Point separated string: Index of the taxa"}}
        self.D_parser["-nf"] = {"name": "--nfolds", "arg": {"metavar": "nfolds",
                                                            "type": int,
                                                            "default": 0,
                                                            "help": "Number of folds for the grid search"}}
        self.D_parser["-mla"] = {"name": "--machine-learning-algorithm",
                                 "arg": {"metavar": "machine_learning_algorithm",
                                         "type": str,
                                         "default": "gmb",
                                         "choices": ["gbm", "dl", "rf", "glm", "aml"],
                                         "help": "Machine learning algorithm used"}}
        self.D_parser["-mm"] = {"name": "--max-models", "arg": {"metavar": "max_models",
                                                                "type": int,
                                                                "default": 1,
                                                                "help": "Number of models trained in the random grid search"}}
        self.D_parser["-prv"] = {"name": "--path-read2vec", "arg": {"metavar": "path_read2vec",
                                                                    "type": str,
                                                                    "default": None,
                                                                    "help": "Complete path to read2vec model"}}
        self.D_parser["-prg"] = {"name": "--path-read2genome", "arg": {"metavar": "path_read2genome",
                                                                       "type": str,
                                                                       "default": None,
                                                                       "help": "Complete path to read2vec model"}}
        self.D_parser["-pmca"] = {"name": "--path-metagenome-cut-analyse",
                                  "arg": {"metavar": "path_metagenome_cut_analyse",
                                          "type": str,
                                          "default": None,
                                          "help": "Complete path to the json file containing two keys 'to_cut' and 'no_cut' with values as a list of metagenome names"}}
        self.D_parser["-pkv"] = {"name": "--path-kmer2vec", "arg": {"metavar": "path_kmer2vec",
                                                                    "type": str,
                                                                    "default": None,
                                                                    "help": "Complete path to kmer2vec model"}}
        self.D_parser["-pt"] = {"name": "--path-tmp-folder", "arg": {"metavar": "path_tmp_folder",
                                                                     "type": str,
                                                                     "default": "./" if "TMP" not in os.environ["TMP"] else os.environ["TMP"],
                                                                     "help": "Complete path to the tmp folder used for the script"}}
        self.D_parser["-ot"] = {"name": "--only-transform", "arg": {"action": "store_true",
                                                                    "help": "If True just compute the reads transforming not training"}}
        self.D_parser["-DO"] = {"name": "--dropout", "arg": {"metavar": "dropout",
                                                             "type": float,
                                                             "default": 0.2,
                                                             "help": "Dropout value"}}
        self.D_parser["-DS"] = {"name": "--deepsets-struct", "arg": {"metavar": "deepsets_struct",
                                                                     "type": str,
                                                                     "default": "200,100,1,1",
                                                                     "help": "Comma separated string, first value is the number of hidden layer for phi network, second number of hidden layer for rho network, third number of layers for phi network, fourth number of layers for rho network"}}
        self.D_parser["-DV"] = {"name": "--vae-struct", "arg": {"metavar": "vae_struct",
                                                                "type": str,
                                                                "default": "40,4,1",
                                                                "help": "Comma separated string, first value is the number of hidden dimension, second number of hidden layer before flatten and third is after flatten"}}

        self.D_parser["-nm"] = {"name": "--n-memory", "arg": {"metavar": "n_memory",
                                                              "type": int,
                                                              "default": 5,
                                                              "help": "Number of memory used in giga octet"}}
        self.D_parser["-TU"] = {"name": "--tuning", "arg": {"action": "store_true",
                                                            "help": "Specify to tune or not the model"}}
        self.D_parser["-FT"] = {"name": "--fine-tuning", "arg": {"action": "store_true",
                                                            "help": "Specify to fine tune or not the model"}}
        self.D_parser["-pfsr"] = {"name": "--path-folder-save-read2genome",
                                  "arg": {"metavar": "path_folder_save_read2genome",
                                          "type": str,
                                          "default": None,
                                          "help": "If the path of the folder is given, it will save the matrix returned by read2genome in order to compute it just once"}}
        self.D_parser["-ib"] = {"name": "--is-bok", "arg": {"action": "store_true",
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
        self.D_parser["-mv"] = {"name": "--max-vectors", "arg": {"metavar": "max_vectors",
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
        self.D_parser['-pjma'] = {"name": "--path-json-modif-abundance", "arg": {"metavar": "path_json_modif_abundance",
                                                                                 "type": str,
                                                                                 "required": False,
                                                                                 "help": "Path to the json containing tax_id: factor where tax_id is the id at the tax_level taxonomic level and factor is the multiplicator to change the original abundance balanced"}}
        self.D_parser["-go"] = {"name": "--giga-octet", "arg": {"metavar": "giga_octet",
                                                                "type": float,
                                                                "default": 1.,
                                                                "help": "Giga octet simulation by sample"}}
        self.D_parser['-pap'] = {"name": "--path-abundance-profile", "arg": {"metavar": "path_abundance_profile",
                                                                             "type": str,
                                                                             "required": True,
                                                                              "help": "If the abundance profile is a file then this profile is replicated n_sample times, if it is a folder then n_sample is replaced by the total number of abundance profiles."}}
        self.D_parser['-sa'] = {"name": "--simulate-abundance", "arg": {"action": "store_true",
                                                                       "help": "If true create a simulation abundance, else only compute abundance balanced"}}
        self.D_parser['-im'] = {"name": "--in-memory", "arg": {"action": "store_true",
                                                                        "help": "Compute in memory (pandas instead of spark"}}
        self.D_parser["-il"] = {"name": "--id-label", "arg": {"metavar": "id_label",
                                                              "type": str,
                                                              "default": None,
                                                              "help": "Comma separated id like id.fasta,group. If given compute only one element and path_metadata is not required."}}
        self.D_parser["-ftd"] = {"name": "--list-file-to-del", "arg": {"metavar": "list_file_to_del",
                                                                       "type": str,
                                                                       "default": None,
                                                                       "required": False,
                                                                       "help": "Comma separated name of file within path_data that we don't want to process."}}
        self.D_parser["-bi"] = {"name": "--bowtie-index", "arg": {"metavar": "bowtie_index",
                                                                  "type": str,
                                                                  "default": None,
                                                                  "required": False,
                                                                  "help": "Bowtie index used for alignment"}}
        self.D_parser["-pp"] = {"name": "--paired-prediction", "arg": {"action": "store_true",
                                                               "help": "Read that are paired will have the same prediction from read2genome by taking the max probability of both reads."}}
        self.D_parser["-cv"] = {"name": "--cross-validation", "arg": {"metavar": "cross_validation",
                                                              "type": int,
                                                              "default": 20,
                                                              "help": "Number of cross validation"}}
        self.D_parser["-tm"] = {"name": "--to-merge", "arg": {"action": "store_true",
                                                              "help": "Merge output files from CAMISIM because anonymized was set to false."}}
        self.D_parser["-d"] = {"name": "--disease", "arg": {"metavar": "disease",
                                                            "type": str,
                                                            "required": True,
                                                            "help": "Comma separated list of the string corresponding to the disease class for each dataset"}}
        self.D_parser["-aa"] = {"name": "--add-abundance", "arg": {"metavar": "add_abundance",
                                                                   "type": str,
                                                                   "default": "no",
                                                                   "choices": ["no", "yes", "only"]}}
        self.D_parser["-AF"] = {"name": "--activation-function", "arg": {"metavar": "activation_function",
                                                                 "type": str,
                                                                 "default": "nn.ReLU",
                                                                 "choices": ["nn.ReLU", "nn.LeakyReLU"],
                                                                 "help": "The activation function used during training."}}
        self.D_parser["-sp"] = {"name": "--spark-conf", "arg": {"metavar": "spark_conf",
                                                                 "type": str,
                                                                 "default": None,
                                                                 "help": "Path to the spark configuration file in yaml format."}}

        self.parser_donwload_metagenomic_data()
        self.parser_bok_split()
        self.parser_bok_merge()
        self.parser_clean_raw_metagenomic_data()
        self.parser_metagenome_kmerization()
        self.parser_genome_kmerization()
        self.parser_create_genome_metadata()
        self.parser_create_simulated_read2genome_dataset()
        self.parser_create_simulated_metagenome2vec_dataset()
        self.parser_create_camisim_config_file()
        self.parser_fastdna()
        self.parser_bok()
        self.parser_metagenome2vec()
        self.parser_deepsets()
        self.parser_vae()
        self.parser_snn()
        
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
                  '-if', '-o', '-mo', '-pd', '-ng', '-sp']:
            parser.add_argument(k, self.D_parser[k]['name'], **self.D_parser[k]['arg'])

    @add_subparser
    def parser_bok_merge(self):
        parser = self.subparsers.add_parser('bok_merge')
        for k in ['-lf', '-pg', '-pd', '-np', '-mo', '-o', '-ng', '-sp']:
            parser.add_argument(k, self.D_parser[k]['name'], **self.D_parser[k]['arg'])

    @add_subparser
    def parser_clean_raw_metagenomic_data(self):
        parser = self.subparsers.add_parser('clean_raw_metagenomic_data')
        for k in ['-pd', '-nsl', '-ps', '-if', '-mo', '-np', '-o', '-pg', '-lf', '-ftd', '-im', '-sp']:
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
    def parser_create_genome_metadata(self):
        parser = self.subparsers.add_parser('create_genome_metadata')
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
                self.D_parser[k]["arg"]["help"] = "Path of the create_genome_metadata output"
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
    # METAGENOME2VEC
    ##############################

    @add_subparser
    def parser_bok(self):
        parser = self.subparsers.add_parser('bok')
        for k in ['-pd', '-k', '-o', '-pmd', '-sp']:
            if k == '-k':
                self.D_parser[k]["arg"]["required"] = True
            if k == '-pmd':
                self.D_parser[k]["arg"][
                    "help"] = "Absolute path to a csv file containing 2 columns : 'id.fasta' (the metagenome's id) and 'group' that is the class of a metagenome"
            if k == '-pd':
                self.D_parser[k]["arg"]["required"] = True
                self.D_parser[k]["arg"][
                    "help"] = "Path were are saved the BoK files for each metagenomes. It is also where is saved the BoK final matrix with metagenome info concatenated"
            parser.add_argument(k, self.D_parser[k]['name'], **self.D_parser[k]['arg'])

    @add_subparser
    def parser_metagenome2vec(self):
        parser = self.subparsers.add_parser('metagenome2vec')
        for k in ['-pd', '-ps', '-pmd', '-prv', '-pt', '-prg', '-nsl', '-T', '-pp', '-o', '-il', '-k', '-sp']:
            if k == '-prg':
                self.D_parser[k]["arg"]["required"] = False
            if k == '-nsl':
                self.D_parser[k]["arg"]["default"] = -1
            if k == '-pmd':
                self.D_parser[k]["arg"][
                    "help"] = "Absolute path to a csv file containing 2 columns : 'id.fasta' (the metagenome's id) and 'group' that is the class of a metagenome"
                self.D_parser[k]["arg"]["required"] = False
                self.D_parser[k]["arg"]["default"] = None
            if k == '-T':
                self.D_parser[k]["arg"]["default"] = 0.0
                self.D_parser[k]["arg"]["type"] = float
                self.D_parser[k]["arg"]["help"] = "Value of the threshold for the read2genome"
            parser.add_argument(k, self.D_parser[k]['name'], **self.D_parser[k]['arg'])

    ##############################
    # Models
    ##############################

    @add_subparser
    def parser_deepsets(self):
        parser = self.subparsers.add_parser('deepsets')
        for k in ['-pd', '-pmd', '-ps', '-dn', '-B', '-S', '-R', '-D', '-TS',
                  '-DO', '-DS', '-ig', '-nm', '-TU', '-I', '-r', '-CL', '-pt', '-d', '-cv']:
            if k == '-dn':
                self.D_parser[k]["arg"]["help"] = "The name of the model to save"
                self.D_parser[k]["arg"]["default"] = "deepsets"
            if k == '-ps':
                self.D_parser[k]["arg"]["help"] = "Complete path to the file where is saved the deepsets models and scores"
                self.D_parser[k]["arg"]["required"] = True
            if k == '-pd':
                self.D_parser[k]["arg"]["help"] = "Complete path to the MIL matrix file"
                self.D_parser[k]["arg"]["required"] = True
            if k == '-I':
                self.D_parser[k]["arg"]["help"] = "Number of model trained when tunning"
                self.D_parser[k]["arg"]["default"] = 10
            parser.add_argument(k, self.D_parser[k]['name'], **self.D_parser[k]['arg'])

    @add_subparser
    def parser_vae(self):
        parser = self.subparsers.add_parser('vae')
        for k in ['-pd', '-pmd', '-ps', '-dn', '-B', '-S', '-R', '-D', '-d', '-ct',
                  '-DO', '-DV', '-ig', '-nm', '-TU', '-I', '-r', '-CL', '-cv', '-TS', '-AF']:
            if k == '-dn':
                self.D_parser[k]["arg"]["help"] = "The name of the model to save"
                self.D_parser[k]["arg"]["default"] = "vae"
            if k == '-ct':
                self.D_parser[k]["arg"]["help"] = "If vae uses variational auto encoder, else if ae uses auto encoder"
                self.D_parser[k]["arg"]["choices"] = ["vae", "ae"]
                self.D_parser[k]["arg"]["default"] = "vae"
                self.D_parser[k]["arg"]["type"] = str
            if k == '-ps':
                self.D_parser[k]["arg"]["help"] = "Complete path to the file where is saved the VAE models and scores"
                self.D_parser[k]["arg"]["required"] = True
            if k == '-pd':
                self.D_parser[k]["arg"]["help"] = "Complete path to the MIL matrix file"
                self.D_parser[k]["arg"]["required"] = True
            if k == '-I':
                self.D_parser[k]["arg"]["help"] = "Number of model trained when tunning"
                self.D_parser[k]["arg"]["default"] = 10
            parser.add_argument(k, self.D_parser[k]['name'], **self.D_parser[k]['arg'])

    @add_subparser
    def parser_snn(self):
        parser = self.subparsers.add_parser('snn')
        for k in ['-pd', '-pmd', '-ps', '-dn', '-B', '-S', '-R', '-D', '-d',
                  '-DO', '-DV', '-ig', '-nm', '-TU', '-I', '-r', '-CL', '-cv', '-TS', '-AF']:
            if k == '-dn':
                self.D_parser[k]["arg"]["help"] = "The name of the model to save"
                self.D_parser[k]["arg"]["default"] = "vae"
            if k == '-ct':
                self.D_parser[k]["arg"]["help"] = "If vae uses variational auto encoder, else if ae uses auto encoder"
                self.D_parser[k]["arg"]["choices"] = ["vae", "ae"]
                self.D_parser[k]["arg"]["default"] = "vae"
                self.D_parser[k]["arg"]["type"] = str
            if k == '-ps':
                self.D_parser[k]["arg"]["help"] = "Complete path to the file where is saved the SNN models and scores"
                self.D_parser[k]["arg"]["required"] = True
            if k == '-pd':
                self.D_parser[k]["arg"]["help"] = "Complete path to the MIL matrix file"
                self.D_parser[k]["arg"]["required"] = True
            if k == '-I':
                self.D_parser[k]["arg"]["help"] = "Number of model trained when tunning"
                self.D_parser[k]["arg"]["default"] = 10
            parser.add_argument(k, self.D_parser[k]['name'], **self.D_parser[k]['arg'])

    ##############################
    ##############################
    ##############################

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
    

if __name__ == "__main__":

    parserCreator = ParserCreator()
    args = parserCreator.parser.parse_args()
    dict_commands_metadata = {"download_metagenomic_data": {"path_log": "download", "log_file": "download_metagenomic_data.log"},
                            "clean_raw_metagenomic_data": {"path_log": "metagenome_preprocessing", "log_file": "clean_raw_metagenomic_data.log"},
                            "bok_split": {"path_log": "metagenome_preprocessing", "log_file": "bok_split.log"},
                            "bok_merge": {"path_log": "metagenome_preprocessing", "log_file": "bok_merge.log"},
                            "create_camisim_config_file": {"path_log": "simulation", "log_file": "create_camisim_config_file.log"},
                            "run_camisim": {"path_log": "simulation", "log_file": "run_camisim.log"},
                            "create_genome_metadata": {"path_log": "simulation", "log_file": "create_genome_metadata.log"},
                            "create_simulated_read2genome_dataset": {"path_log": "simulation", "log_file": "create_simulated_read2genome_dataset.log"},
                            "create_simulated_metagenome2vec_dataset": {"path_log": "simulation", "log_file": "create_simulated_metagenome2vec_dataset.log"},
                            "fastdna": {"path_log": "read2genome", "log_file": "fastdna.log"},
                            "bok": {"path_log": "metagenome2vec", "log_file": "bok.log"},
                            "metagenome2vec": {"path_log": "metagenome2vec", "log_file": "metagenome2vec.log"},
                            "deepsets": {"path_log": "model", "log_file": "deepsets.log"},
                            "snn": {"path_log": "model", "log_file": "snn.log"},
                            "vae": {"path_log": "model", "log_file": "vae.log"},}
    
    command_metadata = dict_commands_metadata[args.command]
    path_log, log_file = os.path.join(SCRIPT_DIR, "logs", command_metadata["path_log"]), command_metadata["log_file"]
    os.makedirs(path_log, exist_ok=True)
    logging.basicConfig(filename=os.path.join(path_log, log_file), level=logging.INFO)
    
    # read the spark conf in yaml file
    spark_conf = {}
    if "spark_conf" in args:
        if args.spark_conf is not None:
            with open(args.spark_conf) as f_spark_conf:
                spark_conf = yaml.safe_load(f_spark_conf)
    
    spark = spark_manager.createSparkSession(args.command, **spark_conf)
    logging.info("Starting {}".format(args.command))

    if args.command == "download_metagenomic_data":
        download_from_tsv_file(args.path_data, args.path_save, args.index_sample_id, args.index_url)

    if args.command == "clean_raw_metagenomic_data":
        os.makedirs(args.path_save, exist_ok=True)
        if not args.is_file:
            for folder in os.listdir(args.path_data):
                logging.info("cleaning metagenome {}".format(os.path.join(args.path_data, folder)))
                preprocess_metagenomic_data(spark, os.path.join(args.path_data, folder), args.path_save, args.n_sample_load, args.num_partitions, args.mode, args.in_memory, args.overwrite)
                spark.catalog.clearCache()
        else:
            preprocess_metagenomic_data(spark, args.path_data, args.path_save, args.n_sample_load, args.num_partitions, args.mode, args.in_memory, args.overwrite)

    if args.command == "bok_split":
        os.makedirs(args.path_save, exist_ok=True)
        if not args.is_file:
            for folder in os.listdir(args.path_data):
                logging.info("Computing bok for metagenome metagenome {}".format(os.path.join(args.path_data, folder)))
                bok_split(spark, os.path.join(args.path_data, folder), args.path_save, args.k_mer_size, args.step, args.mode, args.num_partitions, args.overwrite)
                spark.catalog.clearCache()
        else:
             bok_split(spark, args.path_data, args.path_save, args.k_mer_size, args.step, args.mode, args.num_partitions, args.overwrite)
   
    if args.command == "bok_merge":
        bok_merge(spark, args.path_data, args.nb_metagenome, args.num_partitions, args.mode, args.overwrite)

    if args.command == "create_camisim_config_file":
        create_simulated_config_file(args.n_cpus, args.n_sample_by_class, args.computation_type, args.giga_octet,
                                args.path_tmp_folder, args.path_save, args.path_abundance_profile)

    if args.command == "create_genome_metadata":
        create_genome_metadata(args.path_data, args.path_save, args.path_json_modif_abundance, args.path_metadata, args.simulate_abundance)
    
    if args.command == "create_simulated_read2genome_dataset":
        create_simulated_read2genome_dataset(args.path_fastq_file, args.path_mapping_file, args.path_metadata, args.path_save,
                                            args.valid_size, args.n_sample_load, args.overwrite)
    if args.command == "create_simulated_metagenome2vec_dataset":
        create_simulated_metagenome2vec_dataset(args.path_data, args.path_save, args.overwrite, args.to_merge)

    if args.command == "fastdna":
        fastdna = FastDnaPred()
        fastdna.train(args.path_data, args.k_mer_size, args.embedding_size, args.n_steps, args.learning_rate, args.tax_taken,
                    args.path_kmer2vec, args.path_read2genome, args.path_tmp_folder, args.n_cpus, args.noise, args.max_length)
    
    if args.command == "bok":
        df_metadata = pd.read_csv(args.path_metadata)
        bok.transform(spark, args.k_mer_size, args.path_data, df_metadata, args.overwrite)

    if args.command == "metagenome2vec":
        logging.info("Begin metagenome2vec")
        assert args.id_label is not None or args.path_metadata is not None, "At least one between id_label or path_metadata must be defined."
        logging.info("Load read2vec")
        read2vec = FastDnaEmbed(args.path_read2vec, spark, os.path.join(args.path_tmp_folder, "tmp_fastdna"))
        logging.info("Load read2genome")
        read2genome = FastDnaPred(args.path_read2genome, args.path_tmp_folder)
        if args.path_metadata:
            df_metadata = pd.read_csv(args.path_metadata)
            for i, row in df_metadata.iterrows():
                metagenome_name, target = row["id.subject"], row["group"]
                df_metagenome = spark.read.parquet(os.path.join(args.path_data, metagenome_name))
                embedding.transform(spark, args.path_save, df_metagenome, args.k_mer_size, target, metagenome_name, read2vec, args.n_sample_load,
                        read2genome, threshold=args.threshold, paired_prediction=args.paired_prediction, overwrite=args.overwrite, hc=None, save_read2genome=True)
        else:
            metagenome_name, target = args.id_label.split(',')
            df_metagenome = spark.read.parquet(os.path.join(args.path_data, metagenome_name))
            embedding.transform(args.path_save, df_metagenome, args.k_mer_size, target, metagenome_name, spark, read2vec, args.n_sample_load,
                        read2genome, threshold=args.threshold, paired_prediction=args.paired_prediction, overwrite=args.overwrite, hc=None, save_read2genome=True)
        logging.info("End computation")
        # Add the folder where are saved the output of metagenome2vec to path_data
        # Then aggregate the outputs
        embedding.create_finale_files(os.path.join(args.path_save, string_names.metagenome_embeddings_folder))

    if args.command in ["deepsets", "snn", "vae"]:
        SEED = 1234
        import random
        import torch
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        # Create the path where to save the data
        path_save = args.path_save
        os.makedirs(path_save, exist_ok=True)
        # Define parameters for ray tune
        if args.tuning:
            n_memory = args.n_memory * 1000 * 1024 * 1024  # To convert in giga
            D_resource = {"worker": 1, "cpu": 1, "gpu": 0}
            for resource in args.resources.split(","):
                name, value = resource.split(":")
                D_resource[name] = int(value) if name == "worker" else float(value)
            resources_per_trial = {"cpu": D_resource["cpu"], "gpu": D_resource["gpu"]}
        device = nn_utils.set_device(args.id_gpu)
        cv = args.cross_validation
        test_size = args.test_size
        file_name_parameters = 'best_parameters.json'
        # Load data
        X, y_ = load_several_matrix_for_learning(args.path_data, args.path_metadata, args.disease, model_type=args.command)
        

    if args.command == "deepsets":
        hidden_init_phi_, hidden_init_rho_, n_layer_phi_, n_layer_rho_ = [int(x) for x in args.deepsets_struct.split(",")]

        params = {batch_size: args.batch_size,
                n_epoch: args.n_steps,
                learning_rate: args.learning_rate,
                mil_layer: "attention",
                weight_decay: args.weight_decay,
                hidden_init_phi: hidden_init_phi_,
                hidden_init_rho: hidden_init_rho_,
                n_layer_phi: n_layer_phi_,
                n_layer_rho: n_layer_rho_,
                dropout: args.dropout,
                clip: args.clip}

        # Load data
        output_size = 1 if len(np.unique(y_)) == 2 else len(np.unique(y_))
        average = "binary" if output_size <= 2 else "micro"  # when compute scores, change average if binary or multi class
        multi_class = "raise" if output_size <= 2 else "ovr"
        embed_size = X.shape[1] - 2  # - id_subject and genome

        path_best_parameters = os.path.join(path_save, file_name_parameters)
        path_res_benchmark = os.path.join(path_save, 'benchmark.csv')
        # Tune
        if args.tuning:
            nn_utils.ray_hyperparameter_search(X, y_, args.command, nn_utils.cross_val_score, path_save, embed_size, D_resource, output_size,
                                               num_samples=args.n_iterations, cv=cv, test_size=test_size, device=device, random_seed=SEED)

        # Train and test with cross validation
        best_parameters = nn_utils.load_and_update_parameters(path_best_parameters, params)
        logging.info("Best parameters: {}".format(params))

        # cross val scores
        scores = nn_utils.cross_val_score(X, y_, args.command, params, embed_size, output_size, cv=cv, test_size=test_size, path_model=os.path.join(path_save, "deepsets.pt"), device=device)
        data_manager.write_file_res_benchmarck_classif(path_res_benchmark, args.dataset_name, "deepsets", scores)

    if args.command == "vae":
        hidden_dim_, n_layer_before_flatten_, n_layer_after_flatten_ = [int(x) for x in args.vae_struct.split(",")]

        n_genomes = len(X[string_names.genome_name].drop_duplicates())
        input_dim = (n_genomes, X.shape[1] - 2)  # input dimension of the auto encoder
        path_best_parameters = os.path.join(path_save, file_name_parameters)

        params = {batch_size: args.batch_size,
                  activation_function: args.activation_function,
                  n_epoch: args.n_steps,
                  learning_rate: args.learning_rate,
                  weight_decay: args.weight_decay,
                  hidden_dim: hidden_dim_,
                  n_layer_before_flatten: n_layer_before_flatten_,
                  n_layer_after_flatten: n_layer_after_flatten_,
                  dropout: args.dropout,
                  clip: args.clip}

        # Tune
        if args.tuning:
            nn_utils.ray_hyperparameter_search(X, y_, args.command, nn_utils.cross_val_score, path_save, input_dim, D_resource, 
                                               num_samples=args.n_iterations, cv=cv, test_size=test_size, device=device, random_seed=SEED)

        # Train and test with cross validation
        best_parameters = nn_utils.load_and_update_parameters(path_best_parameters, params)
        logging.info("Best parameters: {}".format(params))

        # cross val scores
        scores = nn_utils.cross_val_score(X, y_, args.command, params, input_dim, cv=cv, test_size=test_size, 
                                          path_model=os.path.join(path_save, "vae.pt"), device=device)
        print(scores)

    if args.command == "snn":
        hidden_dim_, n_layer_before_flatten_, n_layer_after_flatten_ = [int(x) for x in args.vae_struct.split(",")]

        n_genomes = len(X[string_names.genome_name].drop_duplicates())
        input_dim = (n_genomes, X.shape[1] - 2)  # input dimension of the auto encoder
        path_best_parameters = os.path.join(path_save, file_name_parameters)

        params = {activation_function: args.activation_function,
                  batch_size: args.batch_size,
                  n_epoch: args.n_steps,
                  learning_rate: args.learning_rate,
                  weight_decay: args.weight_decay,
                  hidden_dim: hidden_dim_,
                  n_layer_before_flatten: n_layer_before_flatten_,
                  n_layer_after_flatten: n_layer_after_flatten_,
                  dropout: args.dropout,
                  clip: args.clip}

        # Tune
        if args.tuning:
            nn_utils.ray_hyperparameter_search(X, y_, args.command, nn_utils.cross_val_score, path_save, embed_size, output_size, D_resource, 
                                               num_samples=args.n_iterations, cv=cv, test_size=test_size, device=device, random_seed=SEED)
     
        path_best_parameters = os.path.join(path_save, 'best_parameters.json')
        best_parameters = nn_utils.load_and_update_parameters(path_best_parameters, params)
        logging.info("Best parameters: {}".format(params))

        # cross val scores
        scores = nn_utils.cross_val_score(X, y_, args.command, params, input_dim, cv=cv, test_size=test_size,
                                          path_model=os.path.join(path_save, "snn.pt"), device=device)
        print(scores)
    
    logging.info("End computing")

