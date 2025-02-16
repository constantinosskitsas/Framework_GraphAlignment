#from torch_geometric.utils.convert import *
#from torch_geometric.data import NeighborSampler as RawNeighborSampler

# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import copy
import networkx as nx 
import os 
import random as rnd
from random import sample
import json
from networkx.readwrite import json_graph
import  algorithms.GradP.graph_utils
import math
from  algorithms.GradP.data import *
import scipy

'''methods for main alg'''




def cal_degree_dict(G_list, G, layer):
    G_degree = G.degree()
    degree_dict = {}
    degree_dict[0] = {}
    for node in G_list:
        degree_dict[0][node] = {node}
    for i in range(1, layer + 1):
        degree_dict[i] = {}
        for node in G_list:
            neighbor_set = []
            for neighbor in degree_dict[i - 1][node]:
                neighbor_set += nx.neighbors(G, neighbor)
            neighbor_set = set(neighbor_set)
            for j in range(i - 1, -1, -1):
                neighbor_set -= degree_dict[j][node]
            degree_dict[i][node] = neighbor_set
    for i in range(layer + 1):
        for node in G_list:
            if len(degree_dict[i][node]) == 0:
                degree_dict[i][node] = [0]
            else:
                degree_dict[i][node] = node_to_degree(G_degree, degree_dict[i][node])
    return degree_dict
    
def seed_link(seed_list1, seed_list2, G1, G2):
    k = 0
    for i in range(len(seed_list1) - 1):
        for j in range(np.max([1, i + 1]), len(seed_list1)):
            if G1.has_edge(seed_list1[i], seed_list1[j]) and not G2.has_edge(seed_list2[i], seed_list2[j]):
                G2.add_edges_from([[seed_list2[i], seed_list2[j]]])
                k += 1
            if not G1.has_edge(seed_list1[i], seed_list1[j]) and G2.has_edge(seed_list2[i], seed_list2[j]):
                G1.add_edges_from([[seed_list1[i], seed_list1[j]]])
                k += 1
    print('Add seed links : {}'.format(k), end = '\t')
    return G1, G2

def node_to_degree(G_degree, SET):
    SET = list(SET)
    SET = sorted([G_degree[x] for x in SET])
    return SET
    

def clip(x):
    if x <= 0:
        return 0
    else:
        return x


def centrality_choicer(Gs, Gt, bins, alpha) -> str:
    print("------- Calculating centralities ... ------ \n")
    centralities = ['degree', 'betweenness', 'closeness', 'eigenvector', 'katz', 'pagerank']
    cents_s = [dict(nx.degree(Gs)),
               nx.betweenness_centrality(Gs), 
               nx.closeness_centrality(Gs),
               nx.eigenvector_centrality(Gs, max_iter = 500, tol = 1e-8),
               nx.katz_centrality(Gs, alpha = 0.01, normalized = False),
               nx.pagerank(Gs,alpha = 0.85, max_iter = 100)]
    cents_t = [dict(nx.degree(Gt)),
               nx.betweenness_centrality(Gt), 
               nx.closeness_centrality(Gt),
               nx.eigenvector_centrality(Gt, max_iter = 500, tol = 1e-8),
               nx.katz_centrality(Gt, alpha = 0.01, normalized = False),
               nx.pagerank(Gt,alpha = 0.85, max_iter = 100)]
    
    print("------- Calculating our measure ... ------ \n")
    results = []
    for cent_s, cent_t in zip(cents_s, cents_t):        
        cs = [_ for _ in cent_s.values()]
        ct = [_ for _ in cent_t.values()]
        max_range = max(max(cs), max(ct))
        min_range = min(min(cs), min(ct))
        hs, _ = np.histogram(cs, bins, range = (min_range, max_range), density = False)
        ht, _ = np.histogram(ct, bins, range = (min_range, max_range), density = False)  
        pdf_s = hs / np.sum(hs)
        pdf_t = ht / np.sum(ht) 
        results.append(our_measure(pdf_s, pdf_t, alpha))
    results_np = np.array(results)
    
    return centralities[np.argmax(results_np)]
        

def our_measure(pdf_s, pdf_t, alpha):    
    kl = sum(scipy.special.kl_div(pdf_s, pdf_t))
    var = np.var(pdf_s)+np.var(pdf_t)
    
    return kl + alpha * var
    
    
 

''' preproc '''

def preprocessing(G1, G2, alignment_dict):
    '''
    Parameters
    ----------
    G1 : source graph
    G2 : target graph
    alignment_dict : grth dict

    '''
    # shift index for constructing union
    # construct shifted dict
    shift = G1.number_of_nodes()
    G2_list = list(G2.nodes())
    G2_shiftlist = list(idx + shift for idx in list(G2.nodes()))
    shifted_dict = dict(zip(G2_list,G2_shiftlist))
    
    #relable idx for G2
    G2 = nx.relabel_nodes(G2, shifted_dict)
    
    #update alignment dict
    align1list = list(alignment_dict.keys())
    align2list = list(alignment_dict.values())   
    shifted_align2list = [a+shift for a in align2list]
    
    groundtruth_dict = dict(zip(align1list, shifted_align2list))
    groundtruth_dict, groundtruth_dict_reversed = get_reversed(groundtruth_dict)
    
    return G2, groundtruth_dict, groundtruth_dict_reversed


def create_idx_dict_pair(G1,G2,alignment_dict):
    '''
    Make sure that this function is followed after preprocessing dict.

    '''
    
    G1list = list(G1.nodes())
    #G1list.sort()
    idx1_list = list(range(G1.number_of_nodes()))
    #make dict for G1
    idx1_dict = {a : b for b, a in zip(idx1_list,G1list)}

    
    G2list = list(G2.nodes())
    #G2list.sort()
    idx2_list = list(range(G2.number_of_nodes()))
    #make dict for G2
    idx2_dict = {c : d for d, c in zip(idx2_list,G2list)}
    
    return idx1_dict, idx2_dict
    
def normalized_adj(G):
    # make sure ordering has ascending order
    deg = dict(G.degree)
    deg = sorted(deg.items())
    deglist = [math.pow(b, -0.5) for (a,b) in deg]
    degarr = np.array(deglist)
    degarr = np.expand_dims(degarr, axis = 0)
    return degarr.T

def greedy_match(X, G1, G2):
    G1_nodes = list(G1.nodes())
    G2_nodes = list(G2.nodes())
    m, n = X.shape
    x = np.array(X.flatten()).reshape(-1, )
    minSize = min(m, n)
    usedRows = np.zeros(n)
    usedCols = np.zeros(m)
    maxList = np.zeros(minSize)
    row = np.zeros(minSize)
    col = np.zeros(minSize)
    ix = np.argsort(-np.array(x))
    matched = 0
    index = 0
    while (matched < minSize):
        ipos = ix[index]
        jc = int(np.floor(ipos / n))
        ic = int(ipos - jc * n)
        if (usedRows[ic] != 1 and usedCols[jc] != 1):
            row[matched] = G1_nodes[ic]
            col[matched] = G2_nodes[jc]
            maxList[matched] = x[index]
            usedRows[ic] = 1
            usedCols[jc] = 1
            matched += 1
        index += 1;
    row = row.astype(int)
    col = col.astype(int)
    return zip(col, row)



