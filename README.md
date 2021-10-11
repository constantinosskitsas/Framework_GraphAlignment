# **Experimental Analysis of Graph Alignment Algorithms.**

## **Introduction**
The graph alignment problem calls for finding a matching between the nodes of one graph and those of another graph, in a way that they correspond to each other by some fitness measure. Over the last years, several graph alignment algorithms have been proposed and evaluated on diverse datasets and quality measures. Typically, a newly proposed algorithm is compared to previously proposed ones on some specific datasets, types of noise, and quality measures where the new proposal achieves superiority over the previous ones. However, no systematic comparison of the proposed algorithms has been attempted on the same benchmarks. This paper fills this gap by conducting an extensive, thorough, and commensurable evaluation of state-of-the-art graph alignment algorithms. Our results indicate that certain overlooked solutions perform competitively, while there is no one-size-fits-all winner.

## Algorithms

We evaluate nine representative graph-alignement algorithms, and their papers and the original codes are given in the following table.

|   ALGORITHM   |     PAPER     |   CODE   |
|:--------:|:------------:|:--------:|
|  GWL  |  [arXiv'2019](https://arxiv.org/abs/1901.06003)  |  [Python](https://github.com/HongtengXu/gwl)  |
|  CΟΝΕ-ALign   |  [CIKM '20](https://dl.acm.org/doi/10.1145/3340531.3412136)  | [Python](https://github.com/GemsLab/CONE-Align) |
|  Grasp        |    [APWeb-WAIM'2021](https://link.springer.com/chapter/10.1007/978-3-030-85896-4_4)    | [Python](https://github.com/juhuhu/GrASp)      |
|  LREA        |    [WWW '2018](https://dl.acm.org/doi/10.1145/3178876.3186128)    |      [Julia](https://github.com/nassarhuda/lowrank_spectral)      |
|  NSD       |    [IEEE'2012](https://ieeexplore.ieee.org/document/5975146)    | [Julia](https://github.com/nassarhuda/NetworkAlignment.jl/blob/master/src/NSD.jl) |
|  Isorank     |    [PNAS'2008](https://www.pnas.org/content/105/35/12763)    |         [-](http://cb.csail.mit.edu/cb/mna/)       |
|  Regal     |    [CIKM '2018](https://dl.acm.org/doi/10.1145/3269206.3271788)    | [C++/MATLAB](https://github.com/ZJULearning/ssg) |
|  Net-Align        |    [ACM'2013](https://dl.acm.org/doi/10.1145/2435209.2435212)    |[Matlab](https://www.cs.purdue.edu/homes/dgleich/codes/netalign/)      |
|  Klau's        | [APBC '2009](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-10-S1-S59) | [Matlab](https://www.cs.purdue.edu/homes/dgleich/codes/netalign/) |
|  S-GWL        | [arXiv'2019](https://arxiv.org/pdf/1905.07645.pdf) | [Python](https://github.com/HongtengXu/s-gwl) |


## Datasets

Our experiment involves seventeen [real-world datasets](https://github.com/constantinosskitsas/Framework_GraphAlignment/blob/master/data.zip)

Also it involves synthetic graphs generated using the networkx library [synthetic graphs](https://networkx.org/documentation/stable/reference/generators.html).

## Parameters

For the average optimal parameters of each algorithm on all experimental datasets, see the [parameters](https://github.com/constantinosskitsas/Framework_GraphAlignment/blob/master/experiment/__init__.py) page.

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
