# exp=17

# pow=1
# a=0
# while [ $a -lt $exp ]
# do
#     a=$((a+1))
#     pow=$((pow*2))
#     # echo $a
#     # echo $pow
# done

# # load=75  #  10-13, 
# # load=136 #  14 
# # load=137 #  15 
# # load=138 #  16 
# load=139 #  17

#####################

load=161 #  10-10000
pow=10

#####################

args="scaling iters=1 mon=True tmp=[${pow}] load=[${load},${load}]"


# ss=5
ss=1m
# ss=5m

# 10, 11, 12 - 5s
# 13, 14, 15, 16, 17 - 1 min
# 10, 100, 1000, 10000 - 1 min

# sleep $ss
# python3 workexp.py with $args run=[0] 
sleep $ss
python3 workexp.py with $args run=[1]
sleep $ss
python3 workexp.py with $args run=[2]
sleep $ss
python3 workexp.py with $args run=[3]
sleep $ss
python3 workexp.py with $args run=[4]
sleep $ss
python3 workexp.py with $args run=[5]
sleep $ss
python3 workexp.py with $args run=[6]

# python3 workexp.py with real tuned accall noise_type=2
# python3 workexp.py with real tuned accall noise_type=3

