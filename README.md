# The Metagenome2Vec python module is a neural network model used to learn vector representations of DNA sequences 


### pre-requisites
- Docker (https://docs.docker.com/engine/install/)
- Define the METAGENOME2VEC_PATH environment variable which is the path to the metagenome2vec folder
- Dowload CAMISIM (https://github.com/CAMI-challenge/CAMISIM) and NanoSim (https://github.com/bcgsc/NanoSim). Define CAMISIM and NANOSIM environment variables corresponding to the path of the CAMISIM and NanoSim softwares respectively.


### Environment creation

##### Build docker image of metagenome2vec
```bash
cd $METAGENOME2VEC_PATH/Docker/metagenome2vec; make
```
Exemple of command to run the container : 
```bash
docker run -i -d --rm --name=metagenome2vec -v /your/path/:/your/path/ -e METAGENOME2VEC_PATH=$METAGENOME2VEC_PATH -e CAMISIM=$CAMISIM -e NANOSIM=$NANNOSIM maxence27/metagenome2vec:1.0
```

##### Build docker image of CAMISIM
```bash
cd $METAGENOME2VEC_PATH/Docker/CAMISIM; make
```

### Examples to execute scripts

##### Data download
```bash
python $METAGENOME2VEC_PATH/main.py download_metagenomic_data -pd /path/to/file/mydata.tsv -ps /path/to/data -isi 1 -iu 6
```

##### Run data preprocessing (cleaning metagenomic data)
From a folder containing metagenomic samples, it cleans these samples and store them in a new folder.
```bash
python $METAGENOME2VEC_PATH/main.py clean_raw_metagenomic_data -pd /path/to/data -ps /path/to/clean_data -nsl 10000
```

##### Run BOK
```bash
python $METAGENOME2VEC_PATH/main.py bok_split -pd /path/to/data -ps /path/to/bok -k 6

python $METAGENOME2VEC_PATH/main.py bok_merge -pd /path/to/bok
```

##### Run simulation
###### Create metadata
From a folder of genomes, it downloads the metadata information on NCBI and save it in a new folder. The new folder will contain the 2 files "fasta\_metadata.csv" and "fasta\_metadata.csv" which represent the metadata of the genomes except that the second one has abundance columns to be included in CAMISIM, and 1 folder named camisim containing the config files for CAMISIM.
```bash
python $METAGENOME2VEC_PATH/main.py create_df_fasta_metadata -pd /path/to/genomic/data -ps /path/to/metadata
```

###### Create config files for camisim
Creates a init file in camisim/config_files considering the different parameter in the command line and it creates an empty folder in camisim/dataset that will be used to saved the simulated data.
```bash
python $METAGENOME2VEC_PATH/main.py create_camisim_config_file -ps /path/to/simulation/folder -nc 3 -nsc 2 -ct both -pt /tmp -go 1.0 -pap /path/to/abundance_file.tsv
```



###### Run CAMISIM
At this time you should have a folder containing a folder called 'camisim' with these files:
- metadata.tsv: header is "genome_ID \t OTU \t NCBI_ID \t novelty_category".
- genome_to_id.tsv: no header, two columns as "genome_ID \t path_to_fasta_file".
- A tsv file with abundance: no header, two columns as "genome_ID \t abundance". Notice that abundance column must sum to 1 and that this file can also be a folder containing several abundance files.

```bash
docker run --rm --name camisim -dt --memory="4g" --memory-swap="4g" --cpus="4.0" -e METAGENOME2VEC_PATH=$METAGENOME2VEC_PATH -e CAMISIM=$CAMISIM -e NANOSIM=$NANOSIM -v /your/path/:/your/path/ maxence27/camisim:1.0`
docker exec -i camisim python $CAMISIM/metagenomesimulation.py --debug $METAGENOME2VEC_PATH/data/simulation_test/camisim/config_files/illumina_abundance_balanced_species.ini
```
The first line initiate the docker container and the second one run the simulation that simulates metagenomic samples in the folder camisim/dataset/my_folder_in_init_file

###### Create a read2genome / fastdna datasets from CAMISIM output
From the simulated data, it creates a dataframe to train and test the read2genome step.
```bash
python $METAGENOME2VEC_PATH/main.py create_simulated_read2genome_dataset -pfq /path/to/reads.fq.gz -pmf /path/to/reads_mapping.tsv.gz -ps /path/to/save/output -nsl 500000 -pmd /path/to/metadata.csv
```

###### Create a metagenome2vec dataset from CAMISIM output
From the simulated data, it creates a dataframe to train and test the metagenome2vec step.
```bash
python $METAGENOME2VEC_PATH/main.py create_simulated_metagenome2vec_dataset -pd /path/to/simulated/data -ps /path/to/save/output
```

##### Run read2genome with fastDNA
```bash
python $METAGENOME2VEC_PATH/main.py fastdna -k 6 -pd /path/to/reads_fastdna,/path/to/fastdna_labels -nc 3 -prg /path/to/save/read2genome -pkv /path/to/save/read2vec -pt /tmp -S 2 -E 50
```

##### Run metagenome2vec
###### Bag of kmers
```bash
python $METAGENOME2VEC_PATH/main.py bok -pd /path/to/folder/with/bok_files -pmd /path/to/metadata.csv -k 6
```

###### Embeddings
From a preprocessed folder of metagenomic data, it will embed the metagenomes with the read2genome and read2vec models.
```bash
python $METAGENOME2VEC_PATH/main.py metagenome2vec -k 6 -pd /path/to/folder/with/metagenomes/preprocessed/ -ps /path/to/save/ -pmd /path/to/metadata.csv -prv /path/to/read2vec -prg /path/to/read2genome
```


##### Train a neural network classifier model
###### Deepsets
```bash
python $METAGENOME2VEC_PATH/main.py deepsets -pd /path/to/the/data -pmd /path/to/the/metadata -dn name_of_the_dataset -B 1 -S 3 -R 0.001 -d target -TU -cv 3 -TS 0.3 -ps /path/to/the/saved/model
```
###### VAE
```bash
python $METAGENOME2VEC_PATH/main.py vae -pd /path/to/the/data -pmd /path/to/the/metadata -dn name_of_the_dataset -B 1 -S 3 -R 0.001 -d target -TU -cv 3 -TS 0.3 -ps /path/to/the/saved/model
```
###### Siamese Network
```bash
python $METAGENOME2VEC_PATH/main.py snn -pd /path/to/the/data -pmd /path/to/the/metadata -dn name_of_the_dataset -B 1 -S 3 -R 0.001 -d target -TU -cv 3 -TS 0.3 -ps /path/to/the/saved/model
```
