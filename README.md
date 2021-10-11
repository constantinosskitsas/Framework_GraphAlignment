# **Experimental Analysis of Graph Alignment Algorithms.**

## **Introduction**
The graph alignment problem calls for finding a matching between the nodes of one graph and those of another graph, in a way that they correspond to each other by some fitness measure. Over the last years, several graph alignment algorithms have been proposed and evaluated on diverse datasets and quality measures. Typically, a newly proposed algorithm is compared to previously proposed ones on some specific datasets, types of noise, and quality measures where the new proposal achieves superiority over the previous ones. However, no systematic comparison of the proposed algorithms has been attempted on the same benchmarks. This paper fills this gap by conducting an extensive, thorough, and commensurable evaluation of state-of-the-art graph alignment algorithms. Our results indicate that certain overlooked solutions perform competitively, while there is no one-size-fits-all winner.

## Algorithms

we evaluate nine representative graph-alignement algorithms, and their papers and the original codes are given in the following table.

|   ALGORITHM   |     PAPER     |   CODE   |
|:--------:|:------------:|:--------:|
|  GWL  |  [WWW'2011](https://dl.acm.org/doi/abs/10.1145/1963405.1963487)  |  [Python](https://github.com/HongtengXu/gwl)  |
|  Cone-ALign   |  [CVPR'2016](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Harwood_FANNG_Fast_Approximate_CVPR_2016_paper.html)  | [Python](https://github.com/GemsLab/CONE-Align) |
|  Grasp        |    [APWeb-WAIM'2021](https://link.springer.com/chapter/10.1007/978-3-030-85896-4_4)    | [Python](https://github.com/juhuhu/GrASp)      |
|  Low Rank Eigen AlignLREA)        |    [TPAMI'2021](https://ieeexplore.ieee.org/abstract/document/9383170)    |      [Julia](https://github.com/nassarhuda/lowrank_spectral)      |
|  NSD       |    [TKDE'2019](https://ieeexplore.ieee.org/abstract/document/8681160)    | [Julia](https://github.com/nassarhuda/NetworkAlignment.jl/blob/master/src/NSD.jl) |
|  Isorank     |    [NeurIPS'2019](http://harsha-simhadri.org/pubs/DiskANN19.pdf)    |         -        |
|  Regal     |    [arXiv'2016](https://arxiv.org/abs/1609.07228)    | [C++/MATLAB](https://github.com/ZJULearning/ssg) |
|  Net-Align        |    [IEEE T CYBERNETICS'2014](https://ieeexplore.ieee.org/abstract/document/6734715/)    |[Matlab](https://www.cs.purdue.edu/homes/dgleich/codes/netalign/      |
|  Klau's        | [IS'2014](https://www.sciencedirect.com/science/article/abs/pii/S0306437913001300) | [Matlab](https://www.cs.purdue.edu/homes/dgleich/codes/netalign/) |
|  S-GWL        | [arXiv'2019](https://arxiv.org/pdf/1905.07645.pdf) | [Python](https://github.com/HongtengXu/s-gwl) |


## Datasets

Our experiment involves eight [real-world datasets](https://github.com/Lsyhprum/WEAVESS/tree/dev/dataset) popularly deployed by existing works. All datasets are pre-split into base data and query data and come with groundtruth data in the form of the top 20 or 100 neighbors. Additional twelve [synthetic datasets](https://github.com/Lsyhprum/WEAVESS/tree/dev/dataset) are used to test the scalability of each algorithm to the performance of different datasets.

Note that, all base data and query data are converted to `fvecs` format, and groundtruth data is converted to `ivecs` format. Please refer [here](http://yael.gforge.inria.fr/file_format.html) for the description of `fvecs` and `ivecs` format. All datasets in this format can be downloaded from [here](https://github.com/Lsyhprum/WEAVESS/tree/dev/dataset).

## Parameters

For the optimal parameters of each algorithm on all experimental datasets, see the [parameters](https://github.com/Lsyhprum/WEAVESS/tree/dev/parameters) page.

## Usage


How to run experiments :
1) python workexp with scaling : This will run the scalability experiment as in the paper/thesis
2) python workexp with tuning : This will run the tunning experiment as in the paper/thesis
3) python workexp with real_noise: This will run the real graphs experiments as in the paper/thesis :MultiMagna,HighSchool,Voles datasets
4) python workexp with real: This will run the high noise experiments as in the paper/thesis 
5) python workexp with arenasish:This will run the random graph experiment+ arenas dataset as in the paper/thesis
6) python workexp with playground: This will run the low noise experiment as in the paper/thesis
7) 
Keywords can be used to make the experiments more specific or add more functionalities :

seed=[***] will run the experiment with specific randomness, it can be used again to run exactly the same experiment

mall - will run all the possible extraction methods for all the selected aglorithms - JonkerVolgenant,Neirest Neigboor,SortGreedy on cost and/or similarity

run=[...] to choose only specific algorithms to run

iters=[..] to speficy the number of iterations

mon=[True] to return results also for memory and Cpu usage

Load= [..] to load the graphs of a specific run id, from the previusly runned . Every experiment creates a unique id.

accs=[...] to specify the evaluation methods         0-acc,1-EC,2-ICS,3-S3,4-Jacc,5-MNC

plot=[..]

no_disc=True

until_connected=False

noise_type-[..] 1 for One-Way, 2 MultiModal ,3 Two-Way
