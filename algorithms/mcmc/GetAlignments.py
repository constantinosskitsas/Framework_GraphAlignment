import heapq
import itertools

import numpy as np
import scipy.sparse as sps
from scipy.sparse import coo_matrix
from sklearn.neighbors import KDTree
import networkx as nx

def get_counterpart(alignment_matrix, true_alignments,K):
    n_nodes = alignment_matrix.shape[0]

    correct_nodes_hits =[]

    if not sps.issparse(alignment_matrix):
        sorted_indices = np.argsort(alignment_matrix)

    for node_index in range(n_nodes):
        target_alignment = node_index #default: assume identity mapping, and the node should be aligned to itself
        if true_alignments is not None: #if we have true alignments (which we require), use those for each node
            target_alignment = int(true_alignments[node_index])
        if sps.issparse(alignment_matrix):
            row, possible_alignments, possible_values = sps.find(alignment_matrix.todense()[node_index])
            node_sorted_indices = possible_alignments[possible_values.argsort()]
        else:
            node_sorted_indices = sorted_indices[node_index]

        if target_alignment in node_sorted_indices[-K:]:
            correct_nodes_hits.append(node_index)

    return correct_nodes_hits

def AlignmentMatrixHitK(matrix,ans_dict,K=5):
    correct_nodes_hitsK = \
        get_counterpart(matrix,ans_dict,K)
    acc_hitsK = len(correct_nodes_hitsK)/matrix.shape[0]
    return acc_hitsK

def KDTreeAlignmentHitK(emb1, emb2, ans_dict,distance_metric="euclidean",K=5):
    ## 稀疏矩阵对齐
    kd_tree = KDTree(emb2, metric=distance_metric)

    dist, ind = kd_tree.query(emb1, k=K)
    row = np.array([])
    for i in range(emb1.shape[0]):
        row = np.concatenate((row, np.ones(K) * i))
    col = ind.flatten()
    data = np.exp(-dist).flatten()
    sparse_align_matrix = coo_matrix((data, (row, col)), shape=(emb1.shape[0], emb2.shape[0]))
    correct_nodes_hitsK = \
        get_counterpart(sparse_align_matrix,ans_dict,K)
    acc_hitsK = len(correct_nodes_hitsK)/emb1.shape[0]

    return acc_hitsK

def KDTreeAlignmentHit1new(emb1, emb2, ans_dict,distance_metric="euclidean"):
    ## 稀疏矩阵对齐
    kd_tree = KDTree(emb2, metric=distance_metric)

    kd_tree.reset_n_calls()

    dist, ind = kd_tree.query(emb1, k=1)
    ind_list = ind[:, 0]

    train_data_dict = dict(zip(list(range(emb1.shape[0])), ind_list))
    cnt = 0
    for key in ans_dict.keys():
        if ans_dict[key] == train_data_dict[key]:
            cnt += 1
    acc = cnt/len(ans_dict)
    return acc, train_data_dict

def KDTreeAlignmentHit1(emb1, emb2, ans_dict,distance_metric="euclidean"):
    ## 稀疏矩阵对齐
    kd_tree = KDTree(emb2, metric=distance_metric)

    kd_tree.reset_n_calls()

    dist, ind = kd_tree.query(emb1, k=1)
    ind_list = ind[:, 0]

    train_data_dict = dict(zip(list(range(emb1.shape[0])), ind_list))
    cnt = 0
    for key in ans_dict.keys():
        if ans_dict[key] == train_data_dict[key]:
            cnt += 1
    acc = cnt/len(ans_dict)
    return acc

def compute_rank(value, L):
    # value越大排名越靠前，最大的排名，rank=1
    arr = np.array(L)
    bool_array = arr > value  # Returns boolean array
    RANK = float(np.sum(bool_array)) + 1
    return RANK

def MAP(matrix, ans_dict):

    r_rank_list = [1.0/compute_rank(matrix[value,ans_dict[value]], matrix[value,:]) for value in range(matrix.shape[0])]

    map = sum(r_rank_list)/len(r_rank_list)
    return map

def AUC(matrix,ans_dict,negative_num):
    rank_list = [compute_rank(matrix[value,ans_dict[value]], matrix[value,:]) for value in range(matrix.shape[0])]
    avg_rank = sum(rank_list)/len(rank_list)
    auc = (negative_num+1-avg_rank)/negative_num
    return auc

def jaccad(item):
    fenzi = len(set([i for i in item[0] if i in item[1]]))
    fenmu = len(set(item[1]+item[0])) + 1e-12
    return fenzi/fenmu

def MNC(seed_dict, c_g, o_g):
    seed_dict_v = [seed_dict[i] for i in seed_dict.keys()]
    c_num = nx.number_of_nodes(c_g)
    o_num = nx.number_of_nodes(o_g)
    c_nodes = list(range(c_num))
    o_nodes = list(range(o_num))

    c_neighbors = [list(c_g.neighbors(node)) for node in c_nodes]
    o_neighbors = [list(o_g.neighbors(node)) for node in o_nodes]

    pi_c_neighbors = [[seed_dict[i] for i in item if i in seed_dict.keys()] for item in c_neighbors]
    accoording_o_neighbors = [[i for i in item if i in seed_dict_v] for item in o_neighbors]
    items = itertools.product(pi_c_neighbors, accoording_o_neighbors)
    jc_score = np.array(list(map(jaccad, items))).reshape(c_num, o_num)
    average = list(jc_score[c_nodes,[seed_dict[i] for i in c_nodes]])

    return sum(average)/len(average)

def EMNC(pred_dict,e1,e2,g1,g2):
    nodes1 = list(pred_dict.keys())
    nodes2 = [pred_dict[i] for i in nodes1]

    adj1 = nx.to_numpy_array(g1,nodelist=list(range(len(g1))))
    adj2 = nx.to_numpy_array(g2,nodelist=list(range(len(g2))))

    D1 = np.sum(adj1,axis=0)
    D2 = np.sum(adj2,axis=0)


    e1_star = np.diag(1.0/D1) @ adj1 @ e1
    e2_star = np.diag(1.0/D2) @ adj2 @ e2

    e1_hat = e1_star[nodes1]
    e2_hat = e2_star[nodes2]

    scores =np.linalg.norm(e1_hat-e2_hat,axis=-1).tolist()
    # print(scores)
    return sum(scores) / len(scores)