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
```bash
python $METAGENOME2VEC_PATH/main.py create_df_fasta_metadata -pd /path/to/genomic/data -ps /path/to/metadata
```

###### Create config files for camisim
```bash
python $METAGENOME2VEC_PATH/main.py create_camisim_config_file -ps /path/to/simulation/folder -nc 3 -nsc 2 -ct both -pt /tmp -go 1.0 -pap /path/to/abundance_file.tsv
```
The script creates a init file in camisim/config_files and an empty folder in camisim/dataset



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
```bash
python $METAGENOME2VEC_PATH/main.py create_simulated_read2genome_dataset -pfq /to/to/reads.fq.gz -pmf /pat/to/reads_mapping.tsv.gz -ps /path/to/save/output -nsl 500000 -pmd /path/to/metadata.csv
```

###### Create a metagenome2vec dataset from CAMISIM output
```bash
python $METAGENOME2VEC_PATH/main.py create_simulated_metagenome2vec_dataset -pd /path/to/simulated/data -ps /path/to/save/output
```

##### Run read2genome with fastDNA
```bash
python $METAGENOME2VEC_PATH/main.py fastdna -k 6 -pd /path/to/reads_fastdna,/path/to/fastdna_labels -nc 3 -prg /path/to/save/read2genome -pkv /path/to/save/read2vec -pt /tmp -S 2 -E 50
```
