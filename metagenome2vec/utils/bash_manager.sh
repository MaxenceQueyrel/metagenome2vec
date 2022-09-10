#!/bin/bash

function parse_yaml {
   local prefix=$2
   local s='[[:space:]]*' w='[a-zA-Z0-9_]*' fs=$(echo @|tr @ '\034')
   sed -ne "s|,$s\]$s\$|]|" \
        -e ":1;s|^\($s\)\($w\)$s:$s\[$s\(.*\)$s,$s\(.*\)$s\]|\1\2: [\3]\n\1  - \4|;t1" \
        -e "s|^\($s\)\($w\)$s:$s\[$s\(.*\)$s\]|\1\2:\n\1  - \3|;p" $1 | \
   sed -ne "s|,$s}$s\$|}|" \
        -e ":1;s|^\($s\)-$s{$s\(.*\)$s,$s\($w\)$s:$s\(.*\)$s}|\1- {\2}\n\1  \3: \4|;t1" \
        -e    "s|^\($s\)-$s{$s\(.*\)$s}|\1-\n\1  \2|;p" | \
   sed -ne "s|^\($s\):|\1|" \
        -e "s|^\($s\)-$s[\"']\(.*\)[\"']$s\$|\1$fs$fs\2|p" \
        -e "s|^\($s\)-$s\(.*\)$s\$|\1$fs$fs\2|p" \
        -e "s|^\($s\)\($w\)$s:$s[\"']\(.*\)[\"']$s\$|\1$fs\2$fs\3|p" \
        -e "s|^\($s\)\($w\)$s:$s\(.*\)$s\$|\1$fs\2$fs\3|p" | \
   awk -F$fs '{
      indent = length($1)/2;
      vname[indent] = $2;
      for (i in vname) {if (i > indent) {delete vname[i]; idx[i]=0}}
      if(length($2)== 0){  vname[indent]= ++idx[indent] };
      if (length($3) > 0) {
         vn=""; for (i=0; i<indent; i++) { vn=(vn)(vname[i])("_")}
         printf("%s%s%s=\"%s\"\n", "'$prefix'",vn, vname[indent], $3);
      }
   }'
}

num_executors=3
executor_cores=5
driver_memory=10g
executor_memory=5g
driver_memory_overhead=5g
executor_memory_overhead=5g
master="local[*]"

function args()
{
    options=$(getopt -o -h \
    --long num-executors: \
    --long executor-cores: \
    --long driver-memory: \
    --long executor-memory: \
    --long driver-memory-overhead: \
    --long executor-memory-overhead: \
    --long master: \
    --long conf-file: \
    --long help \
    -- "$@")
    [ $? -eq 0 ] || {
        echo "Incorrect option provided"
        exit 1
    }
    eval set -- "$options"
    while true; do
        case "$1" in
        --num-executors)
            shift;
            num_executors=$1
            ;;
        --executor-cores)
            shift;
            executor_cores=$1
            ;;
        --driver-memory)
            shift;
            driver_memory=$1
            ;;
        --executor-memory)
            shift;
            executor_memory=$1
            ;;
        --driver-memory-overhead)
            shift;
            driver_memory_overhead=$1
            ;;
        --executor-memory-overhead)
            shift;
            executor_memory_overhead=$1
            ;;
        --master)
            shift;
            master=$1
            ;;
        --conf-file)
            shift;
            conf_file=$1
            ;;
        --help | -h)
            shift;
            help=true
            break
            ;;
        --)
            shift;
            break
            ;;
        esac
        shift
    done
    if [[ -z "$conf_file" ]] && [[ $help != "true" ]]
    then
      usage
      exit
    fi
}

function usage()
{
    echo "mandatory parameters: --conf-file
Optional parameters: --num-executors, --executor-cores, --driver-memory, --executor-memory, --driver_memory_overhead, --executor-memory-overhead, --help / -h"
}