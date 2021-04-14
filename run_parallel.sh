#!/bin/sh

graphs=$1
cores=$2
args=$3
file=$4
seed=$5

if [ -z "$args" ]
then 
    args="plot=False verbose=False save=False"
fi

if [ ! -z "$file" ]
then 
    args="${args} output_path=results/${file} exist_ok=True"
fi

if [ -z "$seed" ]
then 
    seed=$RANDOM
fi

step=$(((graphs+cores-1)/cores))
for ((i=0;i<cores;i++)); do
    first=$((i*step+1))
    last=$(((i+1)*step))
    python workexp.py with _giter=[$first,$last] seed=$seed $args &
    echo $first $last
    sleep 0.1
done
