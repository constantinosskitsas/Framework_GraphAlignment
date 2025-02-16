#import torch_geometric.utils.convert as cv
#from torch_geometric.data import NeighborSampler as RawNeighborSampler
import pandas as pd
from algorithms.GradP.utils import *
import warnings
import argparse
warnings.filterwarnings('ignore')
import collections
import networkx as nx
import copy 
from sklearn.metrics import roc_auc_score
import os
from  algorithms.GradP.models import *
import numpy as np
import random
import torch
from  algorithms.GradP.data import *
import math
import matplotlib.pyplot as plt
from numpy import linalg as LA
def set_seeds(n):
    seed = int(n)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
#seed = 5
#set_seeds(seed)
#print("set seed:", seed)


def parse_args():
    '''
    Parses the arguments.
    '''
    parser = argparse.ArgumentParser(description="Run myProject.")
    parser.add_argument('--attribute_folder', nargs='?', default='dataset/attribute/')
    parser.add_argument('--data_folder', nargs='?', default='dataset/graph/')
    parser.add_argument('--alignment_folder', nargs='?', default='dataset/alignment/',
                         help="Make sure the alignment numbering start from 0")
    parser.add_argument('--graphname', nargs='?', default='fb-tt', 
    help = "eigenvector, pagerank, betweenness, closeness, khop, katz") 
    parser.add_argument('--centrality', nargs='?', default='eigenvector') 
    parser.add_argument('--mode', nargs='?', default='not_perturbed', help="not_perturbed or perturbed") 
    parser.add_argument('--edge_portion', nargs='?', default=0.05,  help="a param for the perturbation case")  
    parser.add_argument('--att_portion', nargs='?', default=0,  help="a param for the perturbation case")
    
    return parser.parse_args()

#args = parse_args()


#with open("./result.txt", "a") as file:
#    file.write(f"---- dataset:{args.graphname} centrality:{args.centrality}\n")

''' ------------------------ Run Grad-Align -----------------------------  '''


def main(data):
    print("GradP")
    np.random.seed(0)
    #args = parse_args()
    Gt=data['Src']
    Gq=data['Tar']
    G1=nx.from_numpy_array(Gt)
    G2=nx.from_numpy_array(Gq)
    n=G1.number_of_nodes()
    m=G2.number_of_nodes()
    for node in G1.nodes():
        if G1.degree(node) == 0:  # Check if the node has a degree of 0
            G1.add_edge(node, node)
    for node in G2.nodes():
        if G2.degree(node) == 0:  # Check if the node has a degree of 0
            G2.add_edge(node, node)
    attr1 = np.ones((len(G1.nodes),1))
        #feature_extraction1(G1)
    attr2 = np.ones((len(G2.nodes),1))

    idx1_dict = {}
    for i in range(len(G1.nodes)): idx1_dict[i] = i
    idx2_dict = {}
    for i in range(len(G2.nodes)): idx2_dict[i] = i
    alignment_dict=idx2_dict
    alignment_dict_reversed=idx2_dict
    #G1, G2, attr1, attr2, alignment_dict, alignment_dict_reversed, idx1_dict, idx2_dict \
    #    = na_dataloader(args)
    centrality='katz'
    #centrality='khop'
    attr1_aug, attr2_aug = augment_attributes(G1, G2,
                                              attr1, attr2,
                                              num_attr = 1,
                                              version = centrality,     
                                              khop = 1,
                                              normalize = False) 
    attr1_aug, attr2_aug = aug_trimming(attr1_aug, attr2_aug)

    #Checking statistics
#    str_con_portion = struct_consist_checker(G1, G2, alignment_dict)
#    att_con_portion = att_consist_checker(G1, G2, attr1, attr2, idx1_dict, idx2_dict, alignment_dict)
#    feat_avg_diff = feat_diff_checker(attr1_aug, attr2_aug)
    k_hop=2;hid_dim=100;train_ratio=0.0;
    GradAlign1 = GradAlign(G2, G1, attr1, attr2, attr1_aug, attr2_aug,k_hop, hid_dim, alignment_dict, alignment_dict_reversed, \
                                      train_ratio, idx1_dict, idx2_dict, alpha = G2.number_of_nodes() / G1.number_of_nodes(), beta = 1)    
    GradAlign1.run_algorithm()
    seed_list1, seed_list2 = GradAlign1.run_algorithm()
    seed_list2=np.array(seed_list2)
    seed_list1=np.array(seed_list1)
    sorted_indices = np.argsort(seed_list2)

# Reorder list_of_nodes2 using the sorted indices
    list_of_nodes2_sorted = seed_list2[sorted_indices]
    list_of_nodes2_sorted=[]
    #print(len(G1.nodes))
    for i in range(len(G1.nodes)):
        list_of_nodes2_sorted.append(i)
# Reorder list_of_nodes1 with the same indices
    list_of_nodes1_sorted = seed_list1[sorted_indices]
    return list_of_nodes1_sorted#seed_list2#, seed_list2,list_of_nodes1_sorted