# Metagenome2Vec

### The Metagenome2Vec python module is a neural network model used to learn vector representations of DNA sequences 


### pre-requisites
- Docker
- Define the METAGENOME2VEC_PATH environment variable which is the path to the metagenome2vec folder
- Dowload CAMISIM (https://github.com/CAMI-challenge/CAMISIM) and NanoSim (https://github.com/bcgsc/NanoSim)


### Environment creation

Build docker image of metagenome2vec
cd $METAGENOME2VEC_PATH/Docker/metagenome2vec; make

Build docker image of CAMISIM
cd $METAGENOME2VEC_PATH/Docker/CAMISIM; make

### Scripts execution

Data download
bash download_metagenomic_data_from_tsv_file.sh --path-input $METAGENOME2VEC_PATH/data/cirrhosis/download_file.tsv --path-output ~/Documents/tmp/data_cirrhosis/

Run data preprocessing :
bash $METAGENOME2VEC_PATH/script/data_processing/metagenome/clean_raw_data.sh --conf-file $METAGENOME2VEC_PATH/script/data_processing/metagenome/clean_raw_data.yaml

Run BOK
bash $METAGENOME2VEC_PATH/script/data_processing/metagenome/bok_split.sh --conf-file $METAGENOME2VEC_PATH/script/data_processing/metagenome/bok_split.yaml
bash $METAGENOME2VEC_PATH/script/data_processing/metagenome/bok_merge.sh --conf-file $METAGENOME2VEC_PATH/script/data_processing/metagenome/bok_merge.yaml

Run Kmerization
bash $METAGENOME2VEC_PATH/script/data_processing/metagenome/kmerization.sh --conf-file $METAGENOME2VEC_PATH/script/data_processing/metagenome/kmerization.yaml

Run simulation
bash $METAGENOME2VEC_PATH/script/data_processing/simulation/create_camisim_config_file.sh --conf-file $METAGENOME2VEC_PATH/script/data_processing/simulation/create_camisim_config_file.yaml
docker run --rm --name camisim -dt --memory="4g" --memory-swap="4g" --cpus="4.0" -e METAGENOME2VEC_PATH=$METAGENOME2VEC_PATH -e CAMISIM=/home/mqueyrel/Documents/CAMISIM -e NANOSIM=/home/mqueyrel/Documents/NanoSim -v /home/mqueyrel/:/home/mqueyrel/ maxence27/camisim:1.0
docker exec -i camisim bash $METAGENOME2VEC_PATH/script/data_processing/simulation/run_camisim.sh --conf-file $METAGENOME2VEC_PATH/script/data_processing/simulation/run_camisim.yaml
