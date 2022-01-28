# Metagenome2Vec

### The Metagenome2Vec python module is a neural network model used to learn vector representations of DNA sequences 

This repository contains the following directories :

Data download
bash download_metagenomic_data_from_tsv_file.sh --path-input $METAGENOME2VEC_PATH/data/cirrhosis/download_file.tsv --path-output ~/Documents/tmp/data_cirrhosis/

Run data preprocessing :
bash $METAGENOME2VEC_PATH/script/data_processing/metagenome/clean_raw_data.sh --conf-file $METAGENOME2VEC_PATH/script/data_processing/metagenome/clean_raw_data.yaml

Run BOK
bash $METAGENOME2VEC_PATH/script/data_processing/metagenome/bok_split.sh --conf-file $METAGENOME2VEC_PATH/script/data_processing/metagenome/bok_split.yaml
bash $METAGENOME2VEC_PATH/script/data_processing/metagenome/bok_merge.sh --conf-file $METAGENOME2VEC_PATH/script/data_processing/metagenome/bok_merge.yaml

Run Kmerization
bash $METAGENOME2VEC_PATH/script/data_processing/metagenome/kmerization.sh --conf-file $METAGENOME2VEC_PATH/script/data_processing/metagenome/kmerization.yaml

