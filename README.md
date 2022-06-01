# **Experimental Analysis of Graph Alignment Algorithms.**

## **Introduction**
The graph alignment problem calls for finding a matching between the nodes of one graph and those of another graph, in a way that they correspond to each other by some fitness measure. Over the last years, several graph alignment algorithms have been proposed and evaluated on diverse datasets and quality measures. Typically, a newly proposed algorithm is compared to previously proposed ones on some specific datasets, types of noise, and quality measures where the new proposal achieves superiority over the previous ones. However, no systematic comparison of the proposed algorithms has been attempted on the same benchmarks. This paper fills this gap by conducting an extensive, thorough, and commensurable evaluation of state-of-the-art graph alignment algorithms. Our results indicate that certain overlooked solutions perform competitively, while there is no one-size-fits-all winner.

## Algorithms

We evaluate nine representative graph-alignement algorithms, and their papers and the original codes are given in the following table.

|   Algorithm   |     Paper     |   Original Code   |
|:--------:|:------------:|:--------:|
|  GWL  |  [arXiv'2019](https://arxiv.org/abs/1901.06003)  |  [Python](https://github.com/HongtengXu/gwl)  |
|  CΟΝΕ-ALign   |  [CIKM '20](https://dl.acm.org/doi/10.1145/3340531.3412136)  | [Python](https://github.com/GemsLab/CONE-Align) |
|  Grasp        |    [APWeb-WAIM'2021](https://link.springer.com/chapter/10.1007/978-3-030-85896-4_4)    | [Python](https://github.com/juhuhu/GrASp)      |
|  Regal     |    [CIKM '2018](https://dl.acm.org/doi/10.1145/3269206.3271788)    | [Python](https://github.com/GemsLab/REGAL) |
|  LREA        |    [WWW '2018](https://dl.acm.org/doi/10.1145/3178876.3186128)    |      [Julia](https://github.com/nassarhuda/lowrank_spectral)      |
|  NSD       |    [IEEE'2012](https://ieeexplore.ieee.org/document/5975146)    | [Julia](https://github.com/nassarhuda/NetworkAlignment.jl/blob/master/src/NSD.jl) |
|  Isorank     |    [PNAS'2008](https://www.pnas.org/content/105/35/12763)    |         [-](http://cb.csail.mit.edu/cb/mna/)       |
|  Net-Align        |    [ACM'2013](https://dl.acm.org/doi/10.1145/2435209.2435212)    |[Matlab](https://www.cs.purdue.edu/homes/dgleich/codes/netalign/)      |
|  Klau's        | [APBC '2009](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-10-S1-S59) | [Matlab](https://www.cs.purdue.edu/homes/dgleich/codes/netalign/) |
|  S-GWL        | [NeurIPS'2019](https://proceedings.neurips.cc/paper/2019/file/6e62a992c676f611616097dbea8ea030-Paper.pdf) | [Python](https://github.com/HongtengXu/s-gwl) |
| Graal        | [JRSICU'2010](https://royalsocietypublishing.org/doi/10.1098/rsif.2010.0063) | [C](http://www0.cs.ucl.ac.uk/staff/natasa/GRAAL/) |
| Grampa        | [ICML'2020](https://dl.acm.org/doi/abs/10.5555/3524938.3525218) | [-](-) |
| B-Grasp        | [-](-) |[-](-) |



## Datasets

Our experiment involves seventeen [real-world datasets](https://github.com/constantinosskitsas/Framework_GraphAlignment/blob/master/data.zip)
|   Dataset   |     Type     |
|:--------:|:------------:|
|  ca-netscience  | [COLLABORATION NETWORKS](https://networkrepository.com/ca-netscience.php)  |
|  bio-celegans   |     [BIOLOGICAL](https://networkrepository.com/bio-celegans.php) |
|  in-arenas        |        [EMAIL](https://networkrepository.com/email-univ.php)      |
|  inf-euroroad        |            [INFRASTRUCTURE ](https://networkrepository.com/inf-euroroad.php)      |
|  inf-power       |         [INFRASTRUCTURE ](https://networkrepository.com/inf-power.php) |
|  ca-GrQc     | [COLLABORATION ](https://networkrepository.com/ca-GrQc.php) |
|  bio-dmela     |         [BIOLOGICAL](https://networkrepository.com/bio-dmela.php) |
|  ca-AstroPh        | [COLLABORATION ](https://networkrepository.com/ca-AstroPh.php)      |
| soc-hamsterster        |  [Social Networks](https://networkrepository.com/soc-hamsterster.php)
| socfb-Bowdoin47        |             [Facebook ](https://networkrepository.com/socfb-Bowdoin47.php)      |
|  socfb-Hamilton46       |         [Facebook ](https://networkrepository.com/socfb-Hamilton46.php) |
|  socfb-Haverford76     |           [Facebook ](https://networkrepository.com/socfb-Haverford76.php)       |
|  socfb-Swarthmore42       | [Facebook ](https://networkrepository.com/socfb-Swarthmore42.php) |
|  soc-facebook       |    [Social Networks](http://snap.stanford.edu/data/ego-Facebook.html)      |
|  high-school-2013     |           [Social Networks](http://www.sociopatterns.org/datasets/high-school-dynamic-contact-networks/)       |
|  mammalia-voles       | [BIOLOGICAL](https://royalsocietypublishing.org/doi/suppl/10.1098/rsif.2014.1004) |
|  MultiMagna       |    [BIOLOGICAL](https://www3.nd.edu/~cone/multiMAGNA++/)      |


Also it involves synthetic graphs generated using the networkx library [synthetic graphs](https://networkx.org/documentation/stable/reference/generators.html).

## Parameters

For the optimal parameters in terms of accuraccy and running time of each algorithm on all experimental datasets, see the [parameters](https://github.com/constantinosskitsas/Framework_GraphAlignment/blob/master/experiment/__init__.py) page. If running time is not an issue higher embeding dimensionality and more iterations yield better accuracy results.

## Required Libraries
scipy,numpy,networkx,pickle,psutil,matplotlib,sklearn,theano,pymanopt,pandas,pot,pytest,autograd,openpyxl,lapjv(Linux)
## Usage


### How to run experiments :
The following commands generate the relevant figures in our evaluation paper: 
```shell
1)  python workexp with scaling #: This will run the scalability experiment as in the paper
2)  python workexp with tuning #: This will run the tunning experiment as in the paper
3)  python workexp with real_noise #: This will run the real graphs experiments as in the paper :MultiMagna,HighSchool,Voles datasets
4)  python workexp with real #: This will run the high noise experiments as in the paper 
5)  python workexp with arenasish #:This will run the random graph experiment+ arenas dataset as in the paper
6)  python workexp with playground #: This will run the low noise experiment as in the paper
```
### Keywords can be used to make the experiments more specific or add more functionalities :
```shell
seed=[***] # will run the experiment with specific randomness, it can be used again to run exactly the same experiment

mall #- will run all the possible extraction methods for all the selected aglorithms - JonkerVolgenant,Neirest Neigboor,SortGreedy on cost and/or similarity

run=[...] #to choose only specific algorithms to run 0=GWL,1=Cone etc based on the Algorithms table order

iters=[..] #to speficy the number of iterations

mon=[True] #to return results also for memory and Cpu usage

Load= [..] #to load the graphs of a specific run id, from the previusly runned . Every experiment creates a unique id.

accs=[...] #to specify the evaluation methods         0-acc,1-EC,2-ICS,3-S3,4-Jacc,5-MNC

plot=[..] #create a plot

no_disc=True #nodes to be conected or not

until_connected=False #network to be conected or not

noise_type=[..] #1 for One-Way, 2 MultiModal ,3 Two-Way
```
## Reference

Please cite our work in your publications if it helps your research:

```
The paper is under submission. 
```
