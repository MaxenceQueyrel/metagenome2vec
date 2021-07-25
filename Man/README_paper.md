##### Requirements
python 3.7
pip install tensorflow<br/>
pip3 install torch torchvision<br/>
pip install dill<br/>
pip install pyspark<br/>
pip install pandas<br/>
pip install sklearn<br/>
pip install matplotlib<br/>
pip install torchtext<br/>
pip install -f http://h2o-release.s3.amazonaws.com/h2o/latest_stable_Py.html h2o<br/>
pip install h2o_pysparkling_2.4<br/>
pip install 'ray[tune]'<br/>
pip3 install ax-platform <br/>
pip install scikit-bio <br/>
pip install 

##### Define helpful paths
DEEPGENE="/data/projects/deepgene/metagenome2vec/"<br/>
ANALYSE="/data/projects/deepgene/analyses/"<br/>
DATA="/home/mqueyrel/data/"<br/>
LOGS="/home/mqueyrel/deepGene/Pipeline/logs/"

# STRUCTU GENOME CATALOG
### Step 0.1
Take a genome catalog as input and transforms sequences into kmers.<br/>
Input : concatenated genome reference<br/>
Output : kmerized concatenated genome reference<br/>
Example : <br/>
python $DEEPGENE/data_processing/genome/kmerization.py -pd $DATA/genome/sim_db_concatenated_506/sim_db_concatenated_506.fa -ps $DATA/genome/kmerized/genomes_k_6 -k 6 -s 6

# STRUCTU SIMULATED DATA
### Step 0.2.0
Simulate a datasets for read2genome with CAMISIM <br/>
Example : <br/>
python /export/ionas1/icr3_bak/data/projects/simulation/external_programs/CAMISIM/metagenomesimulation.py /home/mqueyrel/script_paper/simulation/mini_config_read2genome.ini

### Step 0.2.1
structuring of the output from step 0.2.0, number of reads taken could be defined <br/>
Example : <br/>
bash $DEEPGENE/data_processing/simulated/clean_output_simulated.sh $DATA/genome/read2genome_dataset/2020.04.15_16.07.15_sample_0/reads/anonymous_reads.fq.gz $DATA/genome/read2genome_dataset/2020.04.15_16.07.15_sample_0/reads/reads_mapping.tsv.gz $DATA/simulated/

### Step 0.2.2
This script will create several matrix of simulated reads based from step 0.2.1:
- Normal matrix with read, ids... with one training set and one validation set
- fastdna format for all of them with species, genus and family id
Example : <br/>
python $DEEPGENE/data_processing/simulated/create_simulated_datasets.py -pd $DATA/simulated/ -nsl 7300000 -vs 0.3 -o

# Structu metagenomic data
### Step 0.3
The script transforms fasta files, it concatenates the ones from a same metagenome, takes all or some reads (a percentage can be given), and creates a parquet file with one column named reads <br/>
Example : <br/>
spark-submit --master local[25] --driver-memory 16g --executor-memory 8g $DEEPGENE/data_processing/metagenomic/metagenomic_preprocessing.py -nsl 1000000 -o -pd /data/db/biobank/microbiome_public_repo/data/pdb_colorectal14 -ps $DATA/microbiome/colorectal14_1M -mo local

# Create BoK for each metagenome and the whole datasets
### Step 0.4
Count kmers in each metagenomes and for the whole base<br/>
Example : <br/>


# KMER2VEC
## FASTTEXT
### Step 1.1
Create kmer embeddings with fasttext algorithm<br/>
input : file of kmerized genomes from step 0.1<br/>
output : embeddings of kmers

Example :<br/>
python $DEEPGENE/Pipeline/kmer2vec/fasttext_genome.py -pd $DATA/genome/kmerized/genomes_k_6_s_1 -pa $ANALYSE -E 300 -S 30 -R 0.05 -w 6 -k 6 -s 1 -gc 506ref -nc 30 <br/>

## ANALYSE KMER EMBEDDINGS
### Step 1.2
Take as input the kmer embeddings from step 1.1 and compute three kinds of analyse<br/>
- t-SNE projection
- Edit distance vs cosine similarity
- Needleman-Wunsch score vs cosine similarity

Example :<br/>
python $DEEPGENE/analyse/analyse_embeddings.py -ni 10000 -ct 3 -pd $DATA/genome/kmerized/genomes_k_6 -pkv $ANALYSE/kmer2vec/genome/506ref/fasttext/k_6_w_6/embSize_300_nStep_30_learRate_0.05 <br/>

# READ2VEC
### Step 2



# READ2GENOME
### Step 3


# METAGENOME2VEC
### Step 4
