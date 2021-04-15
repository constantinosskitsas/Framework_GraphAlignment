#!/bin/bash

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
    python3 workexp.py with _giter=[$first,$last] seed=$((seed)) $args &
    echo $first $last
    sleep 0.1
done

# bash run_parallel.sh 15 3 "exp1 noise_level=0.05 full run=[2,6,7,8]" 005 seed=196355709
# bash run_parallel.sh 15 3 "exp1 noise_level=0.04 full run=[2,6,7,8]" 004 seed=895321488
# bash run_parallel.sh 15 3 "exp1 noise_level=0.03 full run=[2,6,7,8]" 003 seed=821564523
# bash run_parallel.sh 15 3 "exp1 noise_level=0.02 full run=[2,6,7,8]" 002 seed=123
# bash run_parallel.sh 15 3 "exp1 noise_level=0.01 full run=[2,6,7,8]" 001 seed=123
# bash run_parallel.sh 15 3 "exp1 noise_level=0.00 full run=[2,6,7,8]" 000 seed=123