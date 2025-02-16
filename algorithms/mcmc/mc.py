import random
import numpy as np
from  algorithms.mcmc import mmnc_run
from  algorithms.mcmc.mmnc_run import run_mmnc_align,run_immnc_align
import copy
import networkx as nx
import os

#rom dataset.load_alignment_data import load_graph_alignment_datasets

import sys

np.random.seed(0)
random.seed(0)
def create_align_graph(g, remove_rate, add_rate=0.0):
    np.random.seed(0)

    max_deree = max([g.degree[i] for i in g.nodes()])
    edges = list(g.edges())
    nodes = list(g.nodes())
    remove_num = int(len(edges) * remove_rate)
    add_num = int(len(edges) * add_rate)
    random.shuffle(edges)
    random.shuffle(nodes)
    max_iters = (len(edges) + len(nodes)) * 2

    new_g = copy.deepcopy(g)

    r_edges = []
    while remove_num and max_iters:
        candidate_edge = edges.pop()
        if new_g.degree[candidate_edge[0]] > 1 and new_g.degree[candidate_edge[1]] > 1:
            new_g.remove_edge(candidate_edge[0], candidate_edge[1])
            r_edges.append([candidate_edge])
            remove_num -= 1
        max_iters -= 1

    max_iters = (len(edges) + len(nodes)) * 2
    while add_num and max_iters:
        n1 = random.choice(nodes)
        n2 = random.choice(nodes)
        if n1 != n2 and n1 not in new_g.neighbors(n2):
            if new_g.degree[n1] < max_deree - 1 or new_g.degree[n2] < max_deree - 1:
                new_g.add_edge(n1, n2)
                add_num -= 1
        max_iters -= 1
    return new_g
def shuffle_graph(g,features=None,shuffle=True):

    original_nodes = list(g.nodes())
    new_nodes = copy.deepcopy(original_nodes)
    if shuffle:
        random.shuffle(new_nodes)
    original_to_new = dict(zip(original_nodes, new_nodes))
    new_graph = nx.Graph()
    new_graph.add_nodes_from(new_nodes)
    for edge in g.edges():
        new_graph.add_edge(original_to_new[edge[0]], original_to_new[edge[1]])
    if features is not None:
        new_to_original = {original_to_new[i]: i for i in range(nx.number_of_nodes(g))}
        new_order = [new_to_original[i] for i in range(nx.number_of_nodes(g))]
        features = features[new_order,:]



        return new_graph, original_to_new, features
    return new_graph, original_to_new

def read_real_graph(n, name_, _sep = ' '):
    print(f'Making {name_} graph...')
    filename = open(f'{name_}', 'r')
    lines = filename.readlines()
    G = nx.Graph()
    for i in range(n): G.add_node(i)
    nodes_set = set()
    for line in lines:
        u_v = (line[:-1].split(_sep))
        u = int(u_v[0])
        v = int(u_v[1])
        if u!=v:
            nodes_set.add(u)
            nodes_set.add(v)
            G.add_edge(u, v)
    print(len(nodes_set))
    return G 

def read_list(filename):
    list_nodes = []
    with open(filename) as file:
        for line in file:
            linesplit = line[:-1].split(' ')
            list_nodes.append(int(linesplit[0]))
    return list_nodes
def main(data):
    print("mc")
    # NumPy -> PyTorch
    A = data['Src']
    B = data['Tar']
    Q_real=np.arange(0,len(A))
    train_ratio = 0.04
    K_de = 3
    K_nei = 7
    T = 10
    fast_select = False
    np.random.seed(0)
    ans_dict = {}
    for i in range(len(Q_real)): ans_dict[i] = Q_real[i]
    if True:
        metrics = ["hits1"]
        fast_select = False

        list_of_nodes = run_immnc_align(A,B,ans_dict,
                        train_ratio=train_ratio,
                        K_de=K_de,
                        niters=T,
                        rate=train_ratio*0.5,
                        K_nei=K_nei,
                        r_rate=0,
                        metric=metrics,
                        fast=fast_select)
            #print(list_of_nodes)
        #accuracy = np.sum(np.array(Q_real)==np.array(list_of_nodes))/len(Q_real)
        #print(f'My accuracy is {accuracy}.')
        #print(f'My frob is {frob}.')
        #print(len(list_of_nodes))
        return list_of_nodes