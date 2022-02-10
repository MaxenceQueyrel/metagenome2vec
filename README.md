# The Metagenome2Vec python module is a neural network model used to learn vector representations of DNA sequences 


### pre-requisites
- Docker (https://docs.docker.com/engine/install/)
- Define the METAGENOME2VEC_PATH environment variable which is the path to the metagenome2vec folder
- Dowload CAMISIM (https://github.com/CAMI-challenge/CAMISIM) and NanoSim (https://github.com/bcgsc/NanoSim)


### Environment creation

##### Build docker image of metagenome2vec
cd $METAGENOME2VEC_PATH/Docker/metagenome2vec; make

##### Build docker image of CAMISIM
cd $METAGENOME2VEC_PATH/Docker/CAMISIM; make

### Examples to execute scripts

##### Data download
bash download_metagenomic_data_from_tsv_file.sh --path-input $METAGENOME2VEC_PATH/data/cirrhosis/download_file.tsv --path-output ~/Documents/tmp/data_cirrhosis/

##### Run data preprocessing :
bash $METAGENOME2VEC_PATH/script/data_processing/metagenome/clean_raw_data.sh --conf-file $METAGENOME2VEC_PATH/script/data_processing/metagenome/clean_raw_data.yaml

##### Run BOK
bash $METAGENOME2VEC_PATH/script/data_processing/metagenome/bok_split.sh --conf-file $METAGENOME2VEC_PATH/script/data_processing/metagenome/bok_split.yaml
bash $METAGENOME2VEC_PATH/script/data_processing/metagenome/bok_merge.sh --conf-file $METAGENOME2VEC_PATH/script/data_processing/metagenome/bok_merge.yaml

##### Run Kmerization
bash $METAGENOME2VEC_PATH/script/data_processing/metagenome/kmerization.sh --conf-file $METAGENOME2VEC_PATH/script/data_processing/metagenome/kmerization.yaml

##### Run simulation
In a folder you should have a folder called 'camisim' with these files:
- metadata.tsv: header is "genome_ID \t OTU \t NCBI_ID \t novelty_category".
- genome_to_id.tsv: no header, two columns as "genome_ID \t path_to_fasta_file".
- A tsv file with abundance: no header, two columns as "genome_ID \t abundance". Notice that abundance column must sum to 1 and that this file can also be a folder containing several abundance files.

###### Create config files
```bash
bash $METAGENOME2VEC_PATH/script/data_processing/simulation/create_camisim_config_file.sh --conf-file$METAGENOME2VEC_PATH/script/data_processing/simulation/create_camisim_config_file.yaml
```
The script creates a init file in camisim/config_files and an empty folder in camisim/dataset

###### Run simulation
```bash
docker run --rm --name camisim -dt --memory="4g" --memory-swap="4g" --cpus="4.0" -e METAGENOME2VEC_PATH=$METAGENOME2VEC_PATH -e CAMISIM=/home/mqueyrel/Documents/CAMISIM -e NANOSIM=/home/mqueyrel/Documents/NanoSim -v /home/mqueyrel/:/home/mqueyrel/ maxence27/camisim:1.0`
docker exec -i camisim bash $METAGENOME2VEC_PATH/script/data_processing/simulation/run_camisim.sh --conf-file $METAGENOME2VEC_PATH/script/data_processing/simulation/run_camisim.yaml
```
The first line initiate the docker container and the second one run the simulation that simulates metagenomic samples in the folder camisim/dataset/my_folder_in_init_file
