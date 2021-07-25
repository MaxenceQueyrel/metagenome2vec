#!/bin/bash

fastq_file=$1
mapping_file=$2
save_folder=$3
path_metadata=$4
save_file="$save_folder"/reads
mapping_file_save="$save_folder"/mapping_read_genome

save_file=$(echo $save_file | sed 's/\/\//\//g')
mapping_file_save=$(echo $mapping_file_save | sed 's/\/\//\//g')

if [[ -f $save_file ]] ; then
        rm $save_file
fi

if [[ -f $mapping_file_save ]] ; then
        rm $mapping_file_save
fi

if [[ ! -d $save_folder ]] ; then
        mkdir $save_folder
fi

printf "sim_id\t#anonymous_read_id\tread\n" > $save_file

gzip -cd $fastq_file | awk '(NR-1)%4<2' | awk 'NR%2{printf "%s ",$0;next;}1' \
| sed 's/ /\t/' | sed -e "s/^/read2genome\t/" >> $save_file

gunzip < $mapping_file >> $mapping_file_save
awk '!a[$0]++' $mapping_file_save > .tmp ; mv .tmp $mapping_file_save

if [[ ! -z $path_metadata ]] ; then
  python - $mapping_file_save $path_metadata << EOF
import sys
import pandas as pd
path_mapping = sys.argv[1]
path_metadata = sys.argv[2]

df_mapping_read_genome = pd.read_csv(path_mapping, sep="\t")
df_metadata = pd.read_csv(path_metadata)[["genome_id", "NCBI_ID"]]
d = dict(zip(df_metadata.genome_id, df_metadata.NCBI_ID))
df_mapping_read_genome["tax_id"] = df_mapping_read_genome["genome_id"].apply(lambda x: d[x])
df_mapping_read_genome.to_csv(path_mapping, index=False, sep='\t')

EOF

fi
