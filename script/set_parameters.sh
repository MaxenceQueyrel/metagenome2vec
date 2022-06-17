#!/bin/bash

usage()
{
    echo "mandatory parameters : -f, -p"
}

function args()
{
    options=$(getopt -o f:p: \
    -- "$@")
    [ $? -eq 0 ] || {
        echo "Incorrect option provided"
        exit 1
    }
    eval set -- "$options"
    while true; do
        case "$1" in
        -p)
            shift;
            p=$1
            ;;
        -f)
            shift;
            f=$1
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

if [ -z "$p" ] || [ -z "$f" ]
then
  usage
  exit
fi

arr=(${p//,/ })

for elem in "${arr[@]}"
do
  elem=(${elem//=/ })
  var=${elem[0]}
  val=${elem[1]}
  sed -i "s/^$var=.*$/$var=$val/" $f
done


