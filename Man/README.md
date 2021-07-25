##### Requirements
python 3.7
pip install tensorflow-gpu<br/>
pip3 install torch torchvision<br/>
pip install dill<br/>
pip install pyspark<br/>
pip install pandas<br/>
pip install sklearn<br/>
pip install matplotlib<br/>
pip install seaborn<br/>
pip install torchtext<br/>
pip install -f http://h2o-release.s3.amazonaws.com/h2o/latest_stable_Py.html h2o<br/>
pip install h2o_pysparkling_2.4

##### Define helpful paths
DEEPGENE="/home/mqueyrel/deepGene/Pipeline"<br/>
ANALYSE="/data/projects/deepgene/analyses/"<br/>
DATA="/home/mqueyrel/data/"<br/>
LOGS="/home/mqueyrel/deepGene/Pipeline/logs/"

# STRUCTU GENES CATALOG
### Step 0.1
Take a gene catalog as input and transforms reads into kmers.<br/>
It just splits each k nucleotides without overlap.
A script can also be run before to take random gene over all the catalog

Example : <br/>
python $DEEPGENE/data_processing/gene/gene_catalog_kmerization.py -pd $DATA/gene/IGC.fa -k 3 -TS 0.3 <br/>
bash $DEEPGENE/data_processing/gene/take_random_gene_from_catalog.sh $DATA/gene/IGC.fa $DATA/gene/IGC_5M.fa 500000

# STRUCTU SIMULATED DATA
### Step 0.2.1
Create 2 tsv file :
One named "mapping_read_genome" with for columns : #anonymous_read_id, genome_id, tax_id and read_id. It contains all the ids to map simulated reads.<br/>
The other named "reads" with 3 columns : sim_id, read_id and read. It contains the read sequence, its name and the simulation id where come from the read

First argument is the folder input that has been created by the simulation<br/>
Second argument is the name of the read tsv file : sim_id, read_id, read<br/>
Third argument is the name of the mapping tsv file : anonymous_read_id, genome_id, tax_id, read_id<br/>
4th argument is the number of lines taken in each simulated data file

Example : <br/>
bash $DEEPGENE/data_processing/simulated/take_train_valid_simulted_reads.sh /export/ionas1/icr3_bak/data/projects/simulation/datal/illumina/raw_simulated/250.samples.26JUL2019/ $DATA/simulated_new/ 1000000 10000

### Step 0.2.2
This script wile create several matrix of simulated reads based on the previous one:
- Normal matrix with read, ids...
- Balanced
- Unbalanced with same size as balanced one
- fastdna format for all of them with species, genus and family id

python $DEEPGENE/data_processing/simulated/create_simulated_datasets.py -pd $DATA/simulated/ -nsc 10000 -ba both -o

# Structu metagenomic data
### Step 0.3
The script transforms fasta files, it concatenates the ones from a same metagenome, takes all or some reads (a percentage can be given), and creates a parquet file with one column named reads

spark-submit --master local[25] --driver-memory 16g --executor-memory 8g $DEEPGENE/data_processing/metagenomic/metagenomic_preprocessing.py -sa 0.01 -o -t -pd /data/db/biobank/microbiome_public_repo/data/pdb_cirrhosis14 -ps $DATA/microbiome/colorectal14_1p -mo local

# BoK
### Step 0.4
Generate a bok matrix used as basline<br/>
spark-submit --master local[52] --driver-memory 150g --executor-memory 50g --conf spark.driver.maxResultSize=0 $DEEPGENE/data_processing/metagenomic/word_count_split.py -w 6 -k 6 -s 1 -pg $DEEPGENE/logs/wordcount -o -ps $DATA/word_count/colorectal14_1p -np 50 -pd $DATA/microbiome/colorectal14_1p -mo local
spark-submit --master local[52] --driver-memory 150g --executor-memory 50g --conf spark.driver.maxResultSize=0 $DEEPGENE/data_processing/metagenomic/word_count_merge.py -pg $DEEPGENE/logs/wordcount -np 50 -pd $DATA/word_count/colorectal14_1p/k_6_w_6_s_1 -mo local


# KMER2VEC
## FASTTEXT
### Step 1.1
Create kmer embeddings with fasttext algorithm<br/>
input : file of kmerized genes from step 0.1<br/>
output : embeddings of kmers

Example :<br/>
python $DEEPGENE/kmer2vec/fasttext_gene.py -pd $DATA/gene/IGC_k_6_1M_train.txt -pa $ANALYSE -E 300 -S 20 -R 0.05 -w 6 -k 6 -gc IGC -nc 40

## ANALYSE KMER EMBEDDINGS
### Step 1.2
Take as input the kmer embeddings from step 1.1 and compute three kinds of analyse<br/>
- t-SNE projection
- Edit distance vs cosine similarity
- Needleman-Wunsch score vs cosine similarity

Example :<br/>
python deepGene/Pipeline/analyse/analyse_embeddings.py -nc 1 -ni 10000 -ct 3 -pmwc /data/projects/deepgene/data/word_count/colorectal14_1p/k_6_s_1/df_word_count.parquet -pkv /data/projects/deepgene/analyses/kmer2vec/gene/IGC/fasttext/k_6_w_6/embSize_100_nStep_30_learRate_0.05/

# READ2VEC
### Step 2
Use previous kmer embeddings computed at the step 1.1 and create an embeddings of read

Example :
python $DEEPGENE/read2vec/seq2seq.py -pa $ANALYSE -pd $DATA/gene/IGC_k_6_1M -gc IGC -k 6 -w 6 -pal embSize_300_nStep_20_learRate_0.05 -kea fasttext -S 10 -B 64 -E 500 -ig 0,1 -I 1000 -R 0.0005<br/>
python $DEEPGENE/read2vec/seq2seq.py -pa $ANALYSE -pd $DATA/gene/IGC_k_9_1M -k 9 -S 30 -B 128 -E 300 -ig 0,1 -I 1000 -R 0.000001 -Ml 20

# READ2GENOME
### Step 3.1
Use the simulated data to train an H2O model for read classification
spark-submit --master local[25] --driver-memory 50g --executor-memory 10g --conf spark.driver.maxResultSize=0 $DEEPGENE/read2genome/train_read2genome.py -f test_h2o -pa $ANALYSE -ps $DATA/simulated/ -pd $DATA/simulated/reads_genomes_train_balance_both_10000,$DATA/simulated/reads_genomes_valid -rea basic -prv -prv $ANALYSE/kmer2vec/gene/IGC/fasttext/k_6_w_6/embSize_300_nStep_10_learRate_0.05 -pg $LOGS/read2genome -mo local -np 40 -tl species -pm $ANALYSE/read2genome -o -nsl 200000

### Step 3.2
Analyse the model and compute some metrics
spark-submit --master local[50] --driver-memory 150g --executor-memory 60g --conf spark.driver.maxResultSize=0 $DEEPGENE/analyse/analyse_read2genome.py -mo local -pd $DATA/simulated/basic_fasttext_k6_w6_emb300_reads_genomes_train_balance_both_10000,$DATA/simulated/basic_fasttext_k6_w6_emb300_reads_genomes_valid -pg /home/mqueyrel/tmp -pm /data/projects/deepgene/analyses/read2genome/GBM_test2_basic_fasttext_k6_w6_emb300_species -tl species -nc 40

### Step 3.3
Bowtie build and predict
./bowtie-1.2.3-linux-x86_64/bowtie-build ./bowtie-1.2.3-linux-x86_64/genomes/sim_db_concatenated_506.fa ./bowtie-1.2.3-linux-x86_64/indexDir --thread 30
./bowtie-1.2.3-linux-x86_64/bowtie -a -v 1 indexDir --suppress 1,5,6,7 -c ATGCATCATGCGCCAT | grep ">" | wc -l

# METAGENOME2VEC
### Step 4
Transform the original dataset of fastq file to a vectorial representation. <br/>
Can use kmer2vec, read2vec and read2genome

spark-submit --master local[25] --driver-memory 50g --executor-memory 10g --conf spark.driver.maxResultSize=0 $DEEPGENE/metagenome2vec/metagenome2vec.py -pa $ANALYSE -k 6 -w 6 -kea fasttext -pal embSize_300_nStep_20_learRate_0.05 -dn colorectal_1p -pd $DATA/colorectal_1p -pg $LOGS/metagenome2vec -rea basic -gc IGC -pmwc $DATA/learning/cirrhosis14_20p/cirrhosis14_20p/kmer_context/k_6_w_6/df_word_count.parquet -pmi $DEEPGENE/../data/colorectal/vs.csv -nsl 20000 -pm $ANALYSE/read2genome/XGBoost_basic_fasttext_k6_w6_emb300_genus -np 40 -mo local -T 0.3 -ct 0,1,2,3 -t

# ANALYSE
## Custer map
### Step 5.1
Cluster map <br/>
python $DEEPGENE/analyse/cluster_map.py -o -pmi $DEEPGENE/../data/colorectal/vs.csv -pd $ANALYSE/metagenome2vec/colorectal_1p/k_6_w_6-fasttext-embSize_300_nStep_20_learRate_0.05-basic-cut_matrix.csv

## BENCHMARK
### Step 5.2
Benchmark tabular <br/>
python $DEEPGENE/analyse/benchmark_classif.py -pd $ANALYSE/metagenome2vec/colorectal_1p/k_6_w_6-fasttext-embSize_300_nStep_20_learRate_0.05-basic-tabular.csv -ps $DATA/results/benchmark_classif_colorectal_1p.csv -nc 40


## DEEPSETS
### Step 5.3
DeepSets <br/>
python $DEEPGENE/analyse/deepSets.py -pd $ANALYSE/metagenome2vec/colorectal_1p/k_6_w_6-fasttext-embSize_300_nStep_20_learRate_0.05-basic-mil.csv -pm $ANALYSE/deepsets/colorectal_k_6_fasttext_emb_300.pt -B 12 -S 1000 -R 0.001 -MIL attention -TS 0.2 -ig 0 -D 0.0001
### Step 5.4
Grid Search Deep Sets <br/>
python $DEEPGENE/analyse/deepSets_grid_search.py -ig 0 -pd $ANALYSE/metagenome2vec/colorectal_1p/k_6_w_6-fasttext-embSize_300_nStep_20_learRate_0.05-basic-mil.csv -ps ./deepsets_results.csv





