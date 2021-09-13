## Framework_GraphAlignment
Implementation of well-known graph aligment methods in Python.
Work in progress.

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

mall=[True] will run all the possible extraction methods for all the selected aglorithms - JonkerVolgenant,Neirest Neigboor,SortGreedy on cost and/or similarity

runs=[...] to choose only specific algorithms to run

iters=[..] to speficy the number of iterations

mon=[True] to return results also for memory and Cpu usage

Load= [..] to load the graphs of a specific run id, from the previusly runned . Every experiment creates a unique id.

accs=[...] to specify the evaluation methods         0-acc,1-EC,2-ICS,3-S3,4-Jacc,5-MNC

noise_type-[..] 1 for One-Way, 2 MultiModal ,3 Two-Way
