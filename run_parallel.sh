#!/bin/bash

graphs=$1
cores=$2
args=$3
file=$4
log_level=$5
seed=$6
exe=$7

# if [ -z "${args}" ]
# then 
#     args="plot=False save=False verbose=False"
# fi

if [ ! -z "${file}" ]
then 
    args="${args} output_path=results/${file} exist_ok=True"
fi

if [ -z "${log_level}" ]
then 
    log_level="DEBUG"
fi

if [ -z "${seed}" ]
then 
    seed=$RANDOM
fi

if [ -z "${exe}" ]
then 
    exe="python3"
fi



if [ -z "$file" ]
then 
    tmux new -d
else
    tmux new -d -s "${file}"
fi

step=$(((graphs+cores-1)/cores))
for ((i=0;i<cores;i++)); do
    first=$((i*step+1))
    last=$(((i+1)*step))
	giter="[${first},${last}]"

    if [ $i -eq 0 ]
    then
        tmux rename-window "${giter}"
    else
        tmux neww -d -n "${giter}"
    fi


	tmux send-keys -t "${giter}" "${exe} workexp.py -l ${log_level} with _giter=${giter} seed=$((seed)) ${args}" Enter
    # python3 workexp.py with _giter="${giter}" seed=$((seed)) "${args}" &
    echo $first $last
done

# bash run_parallel.sh 4 4 "verbose=False plot=False save=False full" test DEBUG 123 python
# bash run_parallel.sh 4 4 "verbose=False plot=False save=False run=[0,2] mtype=[10,10,10] iters=1" deb


# bash run_parallel.sh 15 3 "exp1 noise_level=0.05 full run=[2,6,7,8]" 005 seed=196355709
# bash run_parallel.sh 15 3 "exp1 noise_level=0.04 full run=[2,6,7,8]" 004 seed=895321488
# bash run_parallel.sh 15 3 "exp1 noise_level=0.03 full run=[2,6,7,8]" 003 seed=821564523
# bash run_parallel.sh 15 3 "exp1 noise_level=0.02 full run=[2,6,7,8]" 002 seed=123
# bash run_parallel.sh 15 3 "exp1 noise_level=0.01 full run=[2,6,7,8]" 001 seed=123
# bash run_parallel.sh 15 3 "exp1 noise_level=0.00 full run=[2,6,7,8]" 000 seed=123

# bash run_parallel.sh 9 3 "exp1 noise_level=0.05 full" 005r
# bash run_parallel.sh 9 3 "exp1 noise_level=0.04 full" 004r
# bash run_parallel.sh 9 3 "exp1 noise_level=0.03 full" 003r
# bash run_parallel.sh 9 3 "exp1 noise_level=0.02 full" 002r
# bash run_parallel.sh 9 3 "exp1 noise_level=0.01 full" 001r
# bash run_parallel.sh 9 3 "exp1 noise_level=0.00 full" 000r

# bash run_parallel.sh 3 3 "exp1 noise_level=0.05 run=[0] mtype=[10] iters=10 GW_args.opt_dict.epochs=10" gwjv005r DEBUG 7415
# bash run_parallel.sh 3 3 "exp1 noise_level=0.04 run=[0] mtype=[10] iters=10 GW_args.opt_dict.epochs=10" gwjv004r DEBUG 7914
# bash run_parallel.sh 3 3 "exp1 noise_level=0.03 run=[0] mtype=[10] iters=10 GW_args.opt_dict.epochs=10" gwjv003r DEBUG 32298
# bash run_parallel.sh 3 3 "exp1 noise_level=0.02 run=[0] mtype=[10] iters=10 GW_args.opt_dict.epochs=10" gwjv002r DEBUG 1644
# bash run_parallel.sh 3 3 "exp1 noise_level=0.01 run=[0] mtype=[10] iters=10 GW_args.opt_dict.epochs=10" gwjv001r DEBUG 4245
# bash run_parallel.sh 3 3 "exp1 noise_level=0.00 run=[0] mtype=[10] iters=10 GW_args.opt_dict.epochs=10" gwjv000r DEBUG 31287