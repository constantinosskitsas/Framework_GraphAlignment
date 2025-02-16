import heapq
import itertools
import json
import os
import random
from time import time

import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize
import pandas as pd
from scipy.sparse import csgraph
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
#import dgl
import networkx as nx
import heapq
import scipy.sparse as sps
from scipy.sparse import coo_matrix
from sklearn.neighbors import KDTree
from  algorithms.mcmc.GetAlignments import KDTreeAlignmentHit1, KDTreeAlignmentHitK, MAP, AUC, MNC, EMNC, KDTreeAlignmentHit1new
from  algorithms.mcmc.tools import evaluate, read_tex_graph, create_align_graph, \
    shuffle_graph, get_top_k_acc, greedy_match, compute_structural_similarity, jaccad, get_graph_degree_feature, \
    print_run_time,cal_degree_dict

from  algorithms.mcmc.DW import netmf
from numpy import linalg as LA

def convertToPermHungarian2(row_ind,col_ind, n, m):
    P= np.zeros((n,m))
    ans = []
    #print(len(row_ind),len(col_ind),n,m)
    for i in range(n):
        P[row_ind[i]][col_ind[i]] = 1
        #print(row_ind[i],col_ind[i])
        if (row_ind[i] >= n) or (col_ind[i] >= m):
            continue
        ans.append((row_ind[i], col_ind[i]))
    return P, ans

def CenaExtractNodeFeature(g,layers):
    g_degree_dict = cal_degree_dict(list(g.nodes()), g, layers)
    g_nodes = [i for i in range(len(g))]
    N1 = len(g_nodes)
    feature_mat = []
    for layer in range(layers + 1):
        L_max = [np.log( np.max(g_degree_dict[layer][x]) + 1) for x in g_nodes]
        L_med= [np.log(np.median(g_degree_dict[layer][x]) + 1) for x in g_nodes]
        L_min=  [np.log( np.min(g_degree_dict[layer][x]) + 1) for x in g_nodes]
        L_75 = [np.log(np.percentile(g_degree_dict[layer][x], 75) + 1) for x in g_nodes]
        L_25 = [np.log( np.percentile(g_degree_dict[layer][x], 25) + 1) for x in g_nodes]
        feature_mat.append(L_max)
        feature_mat.append(L_min)
        feature_mat.append(L_med)
        feature_mat.append(L_75)
        feature_mat.append(L_25)
    feature_mat = np.array(feature_mat).reshape((-1,N1))
    return feature_mat.transpose()
def select_train_nodes1(g1 ,g2 ,prior_sim ,train_ratio=0.03):
    nodes1 = list(range(len(g1)))
    train_num = int(len(g1 ) *train_ratio)


    seed_dict = greedy_match(prior_sim ,"dict")
    jc_sim_score = jc_sim(seed_dict ,g1 ,g2)

    M = prior_sim * jc_sim_score
    # 挑选出每行最大的值，以及对应的对应情况
    index = np.argmax(M ,axis=1)
    value = M[nodes1 ,index]
    items = [(nodes1[i] ,index[i], value[i]) for i in nodes1]
    items = sorted(items ,key=lambda x :x[2] ,reverse=True)
    items_train = items[:train_num]
    train_dict = {item[0] :item[1] for item in items_train}
    return train_dict


def align_embedding(g1,g2 ,nodes1 ,nodes2,K_nei,r_rate,e1=None,e2=None):
    adj1 = nx.to_numpy_array(g1,nodelist=list(range(len(g1))))
    adj2 = nx.to_numpy_array(g2, nodelist=list(range(len(g2))))
    D1 = np.sum(adj1,axis=0)
    D2 = np.sum(adj2,axis=0)
    dim=128
    if(len(g1)<dim):
        dim=len(g1)-1
    if(len(g2)<dim):
        dim=len(g2)-1
    #path1 = r"dataset/{}/Embedding_G1_{}.npy".format(dataname,str(r_rate))
    #path2 = r"dataset/{}/Embedding_G2_{}.npy".format(dataname, str(r_rate))

    if e1 is None:
        if False:#os.path.exists(path1):
            e1 = np.load(path1)
            e2 = np.load(path2)
        else:
            e1 = netmf(sps.csr_matrix(adj1),dim)
            e2 = netmf(sps.csr_matrix(adj2),dim)



    obj = e1[nodes1].T @  e2[nodes2]
    e1_star = e1
    e2_star = e2

    combined_e1 = [e1]
    combined_e2 = [e2]

    tmp1 = sparse.csr_matrix(np.diag(1 / D1))@sparse.csr_matrix(adj1)
    tmp2 = sparse.csr_matrix(np.diag(1 / D2))@sparse.csr_matrix(adj2)

    s = time()
    for i in range(K_nei):
        e1_star = tmp1 @ e1_star
        e2_star = tmp2 @ e2_star
        combined_e1.append(e1_star)
        combined_e2.append(e2_star)
        obj += e1_star[nodes1].T @  e2_star[nodes2]
    e = time()


    obj = obj/K_nei
    u, _, v = np.linalg.svd(obj)
    R = u @ v
    trans_e1 = e1 @ R

    trans_combined_e1 = np.concatenate([item@ R for item in combined_e1],axis=-1)
    combined_e2 = np.concatenate(combined_e2, axis=-1)

    return trans_e1, e2, trans_combined_e1,combined_e2


def select_train_nodes(e1,e2,train_ratio=0.01,distance_metric="euclidean",num_top=1):
    ## 稀疏矩阵对齐
    n_nodes = e1.shape[0]

    kd_tree = KDTree(e2, metric=distance_metric)

    dist, ind = kd_tree.query(e1, k=num_top)
    dist_list = -dist[:,0]
    ind_list = ind[:,0]

    index_l = heapq.nlargest(int(train_ratio*n_nodes), range(len(dist_list)), dist_list.__getitem__)
    train_data_dict = {i: ind_list[i] for i in index_l}


    return train_data_dict


def fast_select_train_nodes(g1,g2,e1,e2,train_ratio=0.01,distance_metric="euclidean",num_top=1,degree_threshold=6):
    n = min(len(g1),len(g2))
    select_nodes1 = [node for node in g1.nodes() if g1.degree[node]>=degree_threshold]
    select_nodes2 = [node for node in g2.nodes() if g2.degree[node] >= degree_threshold]

    index_dict1 = dict(zip(list(range(len(select_nodes1))),select_nodes1))
    index_dict2 = dict(zip(list(range(len(select_nodes2))),select_nodes2))

    new_e1 = e1[select_nodes1]
    new_e2 = e2[select_nodes2]
    # print("rough select nodes from G1:{}".format(len(select_nodes1)))
    # print("rough select nodes from G2:{}".format(len(select_nodes2)))

    kd_tree = KDTree(new_e2, metric=distance_metric)

    dist, ind = kd_tree.query(new_e1, k=num_top)
    dist_list = -dist[:, 0]
    ind_list = ind[:, 0]
    if int(train_ratio * n)>min(len(select_nodes1),len(select_nodes2)):
        num = min(len(select_nodes1),len(select_nodes2))
    else:
        num=int(train_ratio * n)

    index_l = heapq.nlargest(num, range(len(dist_list)), dist_list.__getitem__)
    train_data_dict = {index_dict1[i]: index_dict2[ind_list[i]] for i in index_l}
    return train_data_dict



degree_thresold = 6
@print_run_time
def run_mmnc_align(g1,g2,ans_dict,K_de=3,K_nei=3,metric=[],train_ratio=0.04,r_rate=0,fast=False):

    #
    S = time()

    #path1= r"dataset/{}/{}_G1_degree_feature.npy".format(dataname,dataname)
    #path2 = r"dataset/{}/{}_G2_degree_feature.npy".format(dataname, dataname)
    if False:#os.path.exists(path1):
        e1 = np.load(path1)
        e2 = np.load(path2)
    else:
        e1 = CenaExtractNodeFeature(g1, K_de)
        e2 = CenaExtractNodeFeature(g2, K_de)
        #np.save(path1,e1)
        #np.save(path2,e2)

    E1 = time()
    # print("feature extraction finished:{}".format(E1-S))
    if fast:
        train_dict = fast_select_train_nodes(g1, g2, e1, e2, train_ratio=train_ratio, degree_threshold=degree_thresold)
    else:
        train_dict = select_train_nodes(e1,e2,train_ratio=train_ratio)
    E2 = time()


    nodes1 = list(train_dict.keys())
    nodes2 = list([train_dict[i] for i in nodes1])
    # print("pseudo anchor links extraction finished:{} ".format(E2-E1))
    # Step 2: Align Embedding Spaces
    aligned_embed1,embed2,trans_combined_e1,combined_e2 = align_embedding(g1,g2,nodes1,nodes2,K_nei,r_rate)
    E3 = time()
    # print("alignment embedding finished:{}".format(E3-E2))
    # Step 3: Match Nodes with Similar Embeddings

    if "hits1" in metric:
        Acc = KDTreeAlignmentHit1(aligned_embed1, embed2, ans_dict)
        print("MMNC, acc_hits@1:{}".format(Acc))

    if "hits5" in metric:
        acc = KDTreeAlignmentHitK(aligned_embed1, embed2, ans_dict, K=5)
        print("MMNC, acc_hits@5:{}".format(acc))
    if "AUC" in metric and "MAP" in metric:
        matrix = euclidean_distances(aligned_embed1, embed2)
        matrix = np.exp(-matrix)
        negative_num = int((1 - Acc) * aligned_embed1.shape[0])
        map = MAP(matrix, ans_dict)
        auc = AUC(matrix, ans_dict, negative_num)
        print("MMNC, MAP:{},AUC:{}".format(map, auc))
    if "MNC" in metric:
        aligments = euclidean_distances(aligned_embed1, embed2)
        values = np.argmax(aligments, axis=-1).tolist()
        pred_dict = dict(zip(list(range(len(values))), values))
        mnc = MNC(pred_dict, g1, g2)
        print("MMNC MNC:{}".format(mnc))
    if "EMNC" in metric:
        aligments = euclidean_distances(aligned_embed1, embed2)
        values = np.argmax(aligments, axis=-1).tolist()
        pred_dict = dict(zip(list(range(len(values))), values))
        emnc = EMNC(pred_dict, aligned_embed1, embed2, g1, g2)
        print("MMNC, EMNC:{}".format(emnc))
    #

#@print_run_time
def run_immnc_align(g1,g2,ans_dict,K_de=3,K_nei=3,
                    metric=[],train_ratio=0.04,niters=10,
                    rate=0.01,r_rate=0,fast=False):
    #path1 = r"dataset/{}/{}_G1_degree_feature.npy".format(dataname, dataname)
    #path2 = r"dataset/{}/{}_G2_degree_feature.npy".format(dataname, dataname)
    #A = nx.to_numpy_array(g1)
    #B = nx.to_numpy_array(g2)
    A=g1
    B=g2
    g1=nx.from_numpy_array(A)
    g2=nx.from_numpy_array(B)
    if False:#os.path.exists(path1):
        embed1 = np.load(path1)
        embed2 = np.load(path2)
    else:
        embed1 = CenaExtractNodeFeature(g1, K_de)
        embed2 = CenaExtractNodeFeature(g2, K_de)
        #np.save(path1, embed1)
        #np.save(path2, embed2)


    # train_dict = select_train_nodes(embed1, embed2, train_ratio=train_ratio)
    train_dict = fast_select_train_nodes(g1,g2,embed1,embed2,train_ratio=train_ratio,degree_threshold=degree_thresold )
    nodes1 = list(train_dict.keys())
    nodes2 = list([train_dict[i] for i in nodes1])
    for i in range(niters):
        # Step 2: Align Embedding Spaces
        if i==0:
            aligned_embed1, embed2,trans_combined_e1,combined_e2 = align_embedding(g1, g2, nodes1,nodes2,K_nei,r_rate)

        else:
            aligned_embed1, embed2, trans_combined_e1, combined_e2 = align_embedding(g1, g2, nodes1,
                                                                                 nodes2, K_nei,r_rate,aligned_embed1,embed2)

        # Step 3: Match Nodes with Similar Embeddings
        if fast:
            train_dict = fast_select_train_nodes(g1, g2, trans_combined_e1, combined_e2,
                                                 train_ratio=max(train_ratio + rate * (i + 1), 1.0),
                                                 degree_threshold=degree_thresold)
        else:
            train_dict = select_train_nodes(trans_combined_e1, combined_e2, train_ratio=max(train_ratio+rate*(i+1),0.5))


        nodes1 = list(train_dict.keys())
        nodes2 = list([train_dict[i] for i in nodes1])
    
    list_of_nodes = []
    if "hits1" in metric:
        Acc, train_data_dict = KDTreeAlignmentHit1new(aligned_embed1, embed2, ans_dict)
        #print("iMMNC, acc_hits@1:{}".format(Acc))
        for i in range(len(g1.nodes)):
            list_of_nodes.append(train_data_dict[i])
        #print(list_of_nodes)
    list_of_nodes2_sorted=[]
    #print(len(g1.nodes))
    for i in range(len(g1.nodes)):
        list_of_nodes2_sorted.append(i)
    #for i in range
    return list_of_nodes