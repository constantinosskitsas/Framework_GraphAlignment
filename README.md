# **Comprehensive Evaluation of Algorithms for Unrestricted Graph Alignment**

## **Introduction**
The graph alignment problem calls for finding a matching between the nodes of one graph and those of another graph, in a way that they correspond to each other by some fitness measure. Over the last years, several graph alignment algorithms have been proposed and evaluated on diverse datasets and quality measures. Typically, a newly proposed algorithm is compared to previously proposed ones on some specific datasets, types of noise, and quality measures where the new proposal achieves superiority over the previous ones. However, no systematic comparison of the proposed algorithms has been attempted on the same benchmarks. This paper fills this gap by conducting an extensive, thorough, and commensurable evaluation of state-of-the-art graph alignment algorithms. Our results indicate that certain overlooked solutions perform competitively, while there is no one-size-fits-all winner.
https://openproceedings.org/2023/conf/edbt/paper-202.pdf
## Algorithms

We evaluate nine representative graph-alignment algorithms, and their papers and the original codes are given in the following table.

|   Algorithm   |     Paper     |   Original Code   |Run Id|
|:--------:|:------------:|:--------:|:--------:|
|  GWL  |  [ICML'2019](https://arxiv.org/abs/1901.06003)  |  [Python](https://github.com/HongtengXu/gwl)  |0|
|  CΟΝΕ-ALign   |  [CIKM '20](https://dl.acm.org/doi/10.1145/3340531.3412136)  | [Python](https://github.com/GemsLab/CONE-Align) |1|
|  Grasp        |    [APWeb-WAIM'2021](https://link.springer.com/chapter/10.1007/978-3-030-85896-4_4)    | [Python](https://github.com/juhuhu/GrASp)      |2|
|  Regal     |    [CIKM '2018](https://dl.acm.org/doi/10.1145/3269206.3271788)    | [Python](https://github.com/GemsLab/REGAL) |3|
|  LREA        |    [WWW '2018](https://dl.acm.org/doi/10.1145/3178876.3186128)    |      [Julia](https://github.com/nassarhuda/lowrank_spectral)      |4|
|  NSD       |    [IEEE'2012](https://ieeexplore.ieee.org/document/5975146)    | [Julia](https://github.com/nassarhuda/NetworkAlignment.jl/blob/master/src/NSD.jl) |5|
|  Isorank     |    [PNAS'2008](https://www.pnas.org/content/105/35/12763)    |         [-](http://cb.csail.mit.edu/cb/mna/)       |6|
|  Net-Align        |    [ACM'2013](https://dl.acm.org/doi/10.1145/2435209.2435212)    |[Matlab](https://www.cs.purdue.edu/homes/dgleich/codes/netalign/)      |7|
|  Klau's        | [APBC '2009](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-10-S1-S59) | [Matlab](https://www.cs.purdue.edu/homes/dgleich/codes/netalign/) |8|
|  S-GWL        | [NeurIPS'2019](https://proceedings.neurips.cc/paper/2019/file/6e62a992c676f611616097dbea8ea030-Paper.pdf) | [Python](https://github.com/HongtengXu/s-gwl) |9|
| Grampa        | [ICML'2020](https://dl.acm.org/doi/abs/10.5555/3524938.3525218) | [-](-) |10|
| B-Grasp        | [TKDD'23](https://dl.acm.org/doi/full/10.1145/3561058) |[Python](https://github.com/AU-DIS/GRASP) |11|
| Fugal        | [-](-) |[-](-) |12|
|  FAQ       | [-](-) |[-](-) |13|
| Got        | [NIPS'19](https://arxiv.org/abs/1906.02085) |[Python](https://github.com/Hermina/GOT) |14|
|  Fgot      | [AAAI'22](https://cdn.aaai.org/ojs/20738/20738-13-24751-1-2-20220628.pdf) |[Python](https://github.com/Hermina/fGOT) |15|
|   Parrot      | [WWW'23](https://dl.acm.org/doi/10.1145/3543507.3583357) |[Matlab](https://github.com/zhichenz98/PARROT-WWW23) |16|
|   Path      | [TPAMI'08](https://ieeexplore.ieee.org/document/4641936) |[C](https://projects.cbio.mines-paristech.fr/graphm/) |17|
|   DS++      | [TOG'17](https://dl.acm.org/doi/abs/10.1145/3130800.3130826) |[-](-) |18|
|   MDS      | [ICLR'23](https://arxiv.org/abs/2207.02968) |[Python](https://github.com/BorgwardtLab/JointMDS?tab=readme-ov-file) |19|



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

For the optimal parameters in terms of accuracy and running time of each algorithm on all experimental datasets, see the [parameters](https://github.com/constantinosskitsas/Framework_GraphAlignment/blob/master/experiment/__init__.py) page. If running time is not an issue higher embeding dimensionality and more iterations yield better accuracy results.

## Required Libraries
numpy(1.20.3)
scipy(1.5.2)
networkx(2.5)
pickle,
psutil,
matplotlib(3.3.2)
scikit-learn(0.24),
sacred(0.8.2)
theano (1.0.5)
pymanopt(0.2.5)
pandas(1.1.3)
pot(0.7.0) 
pytest(6.1.1)
autograd (1.3)
openpyxl (3.0.5)
lapjv(Linux) (1.3.14)
fast_pagerank
torch
## Usage


### How to run experiments :
Extract data.zip
The following commands generate the relevant figures in our evaluation paper: 
```shell
1)  python workexp.py with scaling #: This will run the scalability experiment as in the paper
2)  python workexp.py with tuning #: This will run the tunning experiment as in the paper
3)  python workexp.py with real_noise #: This will run the real graphs experiments as in the paper :MultiMagna,HighSchool,Voles datasets
4)  python workexp.py with real #: This will run the high noise experiments as in the paper 
5)  python workexp.py with synthetic #:This will run the random graph experiment+ arenas dataset as in the paper
6)  python workexp.py with playground #: This will run the low noise experiment as in the paper
```

### Example of Small changes in code to edit Experiments :
 If we want to edit experiment "3)real":
 ```shell
 1)We can find the real() in code in Framework_GraphAlignment/experiment/experiment.py ->real().
 2) The function run = [a,b,c,d] we add the algorithm ID we want to evaluate and iter= N the number of repetitions for the experiment
 3) graph_names = [ ] are all the available graphs , many of them are commented , so you can comment/uncomment to keep only the graphs you need to run expeirments
 4) noise_type=1 and noises=[0,0.01,0.02,xxx] we can change also the noise type of the available ones and the noise level for the algorithms.
 5)All these can be changes also by adding keywords when running the experiment ex.  python workexp.py with real noise_type=2
```
### Keywords can be used to make the experiments more specific or add more functionalities :
```shell
seed=[***] # will run the experiment with specific randomness, it can be used again to run exactly the same experiment

mall #- will run all the possible extraction methods for all the selected aglorithms - JonkerVolgenant,Neirest Neigboor,SortGreedy on cost and/or similarity

run=[...] #to choose only specific algorithms to run 0=GWL,1=Cone etc based on the Algorithms table order

iters=[..] #to speficy the number of iterations

mon=[True] #to return results also for memory and Cpu usage

load= [..] #to load the graphs of a specific run id, from the previusly runned . Every experiment creates a unique id.

accs=[...] #to specify the evaluation methods         0-acc,1-EC,2-ICS,3-S3,4-Jacc,5-MNC

plot=[..] #create a plot

no_disc=True #nodes to be conected or not

until_connected=False #network to be conected or not

noise_type=[..] #1 for One-Way, 2 MultiModal ,3 Two-Way

save=true #Store alignment information
```
## Reference

Please cite our work in your publications if it helps your research:

```
@inproceedings{DBLP:conf/edbt/SkitsasOHMK23,
  author    = {Konstantinos Skitsas and
               Karol Orlowski and
               Judith Hermanns and
               Davide Mottin and
               Panagiotis Karras},
  editor    = {Julia Stoyanovich and
               Jens Teubner and
               Nikos Mamoulis and
               Evaggelia Pitoura and
               Jan M{\"{u}}hlig},
  title     = {Comprehensive Evaluation of Algorithms for Unrestricted Graph Alignment},
  booktitle = {Proceedings 26th International Conference on Extending Database Technology,
               {EDBT} 2023, Ioannina, Greece, March 28-31, 2023},
  pages     = {260--272},
  publisher = {OpenProceedings.org},
  year      = {2023},
  url       = {https://doi.org/10.48786/edbt.2023.21},
  doi       = {10.48786/edbt.2023.21},
  timestamp = {Mon, 08 Aug 2022 09:41:38 +0200},
  biburl    = {https://dblp.org/rec/conf/edbt/SkitsasOHMK23.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
## Contact
For any problems or if you want to add your algorithm to the framework contact au647909@uni.au.dk

## Known Problems.
Graph-B works only with LapJV at this moment.
Algorithms from ID 14-19 to be added.
