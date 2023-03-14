# The Metagenome2Vec Python module is a neural network model for learning vector representations of DNA sequences. 


## Requirements
- Install Docker (https://docs.docker.com/engine/install/)
- Clone the git repository:
```bash
git clone https://github.com/MaxenceQueyrel/metagenome2vec.git
```


## Creating the environment

### Build the docker images
##### metagenome2vec image
```bash
cd metagenome2vec
make
```

##### CAMISIM image
```bash
cd metagenome2vec
make build_camisim
```

### How to use the package from the command line
Before running the command lines, the containers can be run from the previous created images.
Below an example to the the containers:
##### metagenome2vec container
```bash
docker run -i -d --rm --name=metagenome2vec -v /local/data/path/:/container/data/path/ maxence27/metagenome2vec:2.0
```
##### CAMISIM container
```bash
docker run --rm --name camisim -dt --memory="4g" --memory-swap="4g" --cpus="4.0" -v /local/data/path/:/container/data/path/ maxence27/camisim:1.0
```

##### Downloading Metagenomic Data
Download the metagenomic fastq files stored in subfolders (one subfolder per sample) in /path/to/data_saved. They are downloaded from the tsv file /path/to/mydata.tsv obtained from the NCBI website, which contains the following columns
study\_accession, sample\_accession, experiment\_accession, run\_accession, tax\_id, scientific\_name, fastq\_ftp, submitted\_ftp, sra\_ftp.


```bash
docker exec -i metagenome2vec python main.py download_metagenomic_data --path-data /path/to/mydata.tsv --path-save /path/to/data_saved --index-sample-id 1 --index-url 6
```

- path_data: Path to the tsv file.
- path_save: Path where the metagenomic data will be written.
- index\_sample\_id: Index in the tsv file of the column containing the sample ids.
- index\_url: Index in the tsv file of the column containing the sample URL.



##### Performing Data Processing (Formatting Metagenomic Data)
From a folder containing metagenomic samples in fastq format, it processes these samples and saves them in a new folder in parquet format (again, one subfolder per metagenome).

```bash
docker exec -i metagenome2vec python main.py clean_raw_metagenomic_data --path-data /path/to/data --path-save /path/to/formated_data
```


##### Running a Simulation

###### Step 1: Create Metadata
```bash
docker exec -i metagenome2vec python main.py create_df_fasta_metadata --path-data /path/to/genomic/data_folder --path-save /path/to/saving_folder
```
- path-data: Path to the folder containing the genomic data files.
- path-save: Path to the folder where the metadata files will be saved.

Given a folder of genomes fasta files, it downloads the metadata information of the genomes from NCBI and saves it in a new folder. The new folder will contain a folder and a file, the folder is named "camisim" containing the configuration files for CAMISIM and the file is named "fasta\_metadata.csv" which represents the metadata of the genomes and the abundance columns to be included in CAMISIM.

###### Step 2: Create config files for camisim
```bash
docker exec -i metagenome2vec python main.py create_camisim_config_file --path-save /path/to/simulation_folder --n-cpus 3 --n-sample-by-class 2 --computation-type both --path-tmp /tmp --giga_octet 1.0 --path-abundance-profile /path/to/abundance_file.tsv
```
- path_save: The same path as in the previous step, corresponding to the path where the configuration files and dataset folders are stored. 
- path_tmp: Path to the tmp folder used by CAMISIM to simulate data.
- n_cpus: Number of CPUs used during the simulation
- n\_sample\_by\_class: The number of samples generated by class (for a given abundance profile)
- computation_type: Can be "camisim", "nanosim" or "both" to create a simulation config file to simulate with CAMISIM, NanoSim or both. If both, 2 configuration files will be created and 2 simulations must be run.
- giga_octet: The size of the simulated metagenomes in giga octet.
- path\_abundance\_profile: Corresponds to the path to the abundance profiles. If it is a file only one profile will be simulated, if it is a folder it must contain multiple abundance profiles. 

Creates an init file in camisim/config_files taking into account the different parameters in the command line and creates empty folder(s) in camisim/dataset where the simulated data will be stored.


###### Step 3: Run CAMISIM
At this point you should have a folder called 'camisim' containing these files:
- metadata.tsv: header is "genome_ID \t OTU \t NCBI_ID \t novelty_category".
- genome_to_id.tsv: no header, two columns as "genome_ID \t path_to_fasta_file".
- A tsv file with abundance: no header, two columns as "genome_ID \t abundance". Note that the abundance column must sum to 1 and that this file can also be a folder containing multiple abundance files.

```bash
docker exec -i camisim python /opt/camisim/metagenomesimulation.py --debug /path/to/save_folder/camisim/config_files/config_file.ini
```
The first line initiates the docker container and the second runs the simulation that simulates the metagenomic samples in the camisim/dataset/my_folder_in_init_file folder.

###### Step4: Create a read2genome / fastdna dataset from the CAMISIM output
Create a dataframe from the simulated data to train and test the read2genome step.
```bash
docker exec -i metagenome2vec python main.py create_simulated_read2genome_dataset --path-fastq-file /path/to/anonymous_reads.fq.gz --path-mapping-file /path/to/reads_mapping.tsv.gz --path_save /path/to/save/output --n-sample-load 500000 --path-metadata /path/to/metadata.csv
```
- path_fastq_file: The path to the simulated fastq file in gunzip format.
- path_mapping_file: The path to the mapping file in gunzip format.
- path_save: The path where the new read2genome data set will be saved. Two files "reads" and "mapping_reads_genomes" are created, the first containing all simulated reads in the dataset and the second containing the mapping between the reads and the genome (the class). In addition, several files are created to enable fastdna training. "reads_fastdna" and "reads_fastdna_valid" refer to the files containing the reads used to train and validate the fastdna model, respectively. The "<taxonomic_level>_fastna" files refer to the files containing the classes of reads used to train and validate the fastdna model at a specific taxonomic level. 
- path_metadata: The path to the metadata used for the simulation.
- n_sample_load: The number of reads to load into the dataset.


###### Step 4 bis: Create a metagenome2vec dataset from the CAMISIM output
Create a dataframe from the simulated data to train and test the metagenome2vec step.
```bash
docker exec -i metagenome2vec python main.py create-simulated-metagenome2vec-dataset --path-data /path/to/simulated/data --path-save /path/to/save/output
```
- path_data: Path to the folder containing all the generated metagenomes. For example, if CAMISIM was asked to simulate 10 metagenomes, then in the output folder there should be 10 folders with the date and the mention of "sample\_0", "sample\_1", etc. in the name.
- path_save: Path where the new data set will be saved. For each sample a specific folder will be created inside the fastq file of the sample and a "metadata.csv" file will also be created to associate the sample with its class (the name of the simulation it comes from).



##### Run read2genome with fastDNA

```bash
docker exec -i metagenome2vec python main.py fastdna --k-mer-size 6 --path-data /path/to/reads_fastdna,/path/to/fastdna_labels --n-cpus 3 --path-read2genome /path/to/save/read2genome --path-kmer2vec /path/to/save/kmer2vec --path-tmp /tmp --n-step 2 --embedding-size 50
```
- k_mer_size: The size of the k-mer used to train the model.
- n_cpus: The number of CPUs used to train the model.
- n_steps: The number of steps used to train the model.
- embedding_size: The size of the trained embeddings.
- path_data: Two paths separated by commas, the first path corresponds to the reads and the second to the classes. These files were created in step 4.
- path_read2genome: The path where the fastdna model is stored.
- path_kmer2vec: The path where the embeddings of the kmers are stored. 
- path_tmp: The path where the temporary data will be stored.


##### Run metagenome2vec
From a preprocessed folder of metagenomic data, it will embed the metagenomes with the read2genome and read2vec models.
```bash
docker exec -i metagenome2vec python main.py metagenome2vec -k 6 -pd /path/to/folder/with/metagenomes/preprocessed/ -ps /path/to/save/ -pmd /path/to/metadata.csv -prv /path/to/read2vec -prg /path/to/read2genome
```


##### Train a neural network classifier model
###### Deepsets
```bash
docker exec -i metagenome2vec python main.py deepsets -pd /path/to/the/data -pmd /path/to/the/metadata -dn name_of_the_dataset -B 1 -S 3 -R 0.001 -d target -TU -cv 3 -TS 0.3 -ps /path/to/the/saved/model
```
###### VAE
```bash
docker exec -i metagenome2vec python main.py vae -pd /path/to/the/data -pmd /path/to/the/metadata -dn name_of_the_dataset -B 1 -S 3 -R 0.001 -d target -TU -cv 3 -TS 0.3 -ps /path/to/the/saved/model
```
###### Siamese Network
```bash
docker exec -i metagenome2vec python main.py snn -pd /path/to/the/data -pmd /path/to/the/metadata -dn name_of_the_dataset -B 1 -S 3 -R 0.001 -d target -TU -cv 3 -TS 0.3 -ps /path/to/the/saved/model
```
