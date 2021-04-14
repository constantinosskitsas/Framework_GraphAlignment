#!/bin/sh

graphs=$1
cores=$2
file=$3
seed=$4

if [ -z $seed ]
then 
    seed=$RANDOM
fi

args="plot=False verbose=False save=False seed=${seed}"

if [ ! -z $file ]
then 
    args="${args} output_path=results/${file} exist_ok=True"
fi


step=$(((graphs+cores-1)/cores))
for ((i=0;i<cores;i++)); do
    first=$((i*step+1))
    last=$(((i+1)*step))
    python workexp.py with _giter=[$first,$last] $args &
    echo $first $last
    sleep 0.1
done
