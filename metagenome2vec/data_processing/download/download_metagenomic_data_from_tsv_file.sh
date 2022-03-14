#! /bin/bash

usage()
{
    echo "mandatory parameters: --path-input, --path-output"
}

index_sample_id=1
index_url=10

function args()
{
    options=$(getopt \
    --long path-input: \
    --long path-output: \
    --long index-sample-id: \
    --long index-url: \
    -- "$@")
    [ $? -eq 0 ] || {
        echo "Incorrect option provided"
        exit 1
    }
    eval set -- "$options"
    while true; do
        case "$1" in
        --path-input)
            shift;
            path_input=$1
            ;;
        --path-output)
            shift;
            path_output=$1
            ;;
        --index-sample-id)
            shift;
            index_sample_id=$1
            ;;
        --index-url)
            shift;
            index_url=$1
            ;;
        --)
            shift
            break
            ;;
        esac
        shift
    done
}

args $0 "$@"

if [ -z "$path_input" ] || [ -z "$path_output" ]
then
  usage
  exit
fi

cpt=1
mkdir -p $path_output

sed 1d $path_input | while IFS=$'\t' read -r -a row
do
  sample_id="${row[$index_sample_id]}"
  echo "Iteration: $cpt, sample_id: $sample_id"
  mkdir -p $path_output/$sample_id
  a_link=(${row[$index_url]//;/ })
  for link in "${a_link[@]}"
  do
    wget ${link} -P $path_output/$sample_id
  done
  cpt=$((cpt+1))
done

