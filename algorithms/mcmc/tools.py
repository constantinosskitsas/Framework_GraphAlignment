import itertools
import math
import time

import numpy as np
import networkx as nx
import random
import copy
import pandas as pd
import torch
import tqdm






# 获取邻接矩阵
def get_adj_matrix(g):
    adj = nx.to_numpy_array(g, nodelist=list(range(len(g))))

    return adj


# greedy search
def greedy_match(S,type="dict"):
    """
    :param S: Scores matrix, shape MxN where M is the number of source nodes,
        N is the number of target nodes.
    :return: A dict, map from source to list of targets.
    """
    S = S.T
    m, n = S.shape
    x = S.T.flatten()
    min_size = min([m,n])
    used_rows = np.zeros((m))
    used_cols = np.zeros((n))
    max_list = np.zeros((min_size))
    row = np.zeros((min_size))  # target indexes
    col = np.zeros((min_size))  # source indexes

    ix = np.argsort(-x) + 1

    matched = 1
    index = 1
    while(matched <= min_size):
        ipos = ix[index-1]

        jc = int(np.ceil(ipos/m))
        ic = ipos - (jc-1)*m
        if ic == 0: ic = 1
        if (used_rows[ic-1] == 0 and used_cols[jc-1] == 0):
            row[matched-1] = ic - 1
            col[matched-1] = jc - 1
            max_list[matched-1] = x[index-1]
            used_rows[ic-1] = 1
            used_cols[jc-1] = 1
            matched += 1
        index += 1
    if type =="dict":
        result = {}
        for i in range(len(row)):
            result[int(col[i])] =  int(row[i])
        return result
    else:
        result = np.zeros(S.T.shape)
        for i in range(len(row)):
            result[int(col[i]), int(row[i])] = 1
        return result


# evaluate
def evaluate(sim, ans_dict,query_nodes=None,type_="greedy"):

    # preds_dict = get_prediction_alignment(e1,e2)
    if type_ == "greedy":

        preds_dict = greedy_match(sim)

    else:
        tmp = np.argmax(sim,axis=1)
        preds_dict = {i:tmp[i] for i in range(sim.shape[0])}
    acc = 0.0
    cnt = 0.0
    if query_nodes is None:
        for key in ans_dict.keys():
            cnt += 1.0

            if preds_dict[key] == ans_dict[key]:
                acc += 1.0
    else:
        for key in ans_dict.keys():
            if key in query_nodes:
                continue
            cnt += 1.0

            if preds_dict[key] == ans_dict[key]:

                acc += 1.0
    return acc/cnt, preds_dict


# top-k acc
def get_top_k_acc(matrix,ans_dict):
    c = [1,5,10,15,20]

    # top_k= dict(zip(c,[0,0,0,0,0]))
    top_k = {1:0,5:0,10:0,15:0,20:0}
    matrix = np.asarray(matrix).astype(np.float)
    index = np.argsort(matrix)


    for j in range(index.shape[0]):
        for i in c:

            if j in ans_dict.keys() and ans_dict[j] in index[j,-i:].tolist():
                top_k[i] += 1

    acc_dict = {i:top_k[i]/len(ans_dict.keys()) for i in top_k.keys()}
    print(acc_dict)


# create graph alignment dataset
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

def remove_nodes(g,remove_num=50):
    pass


# 打乱数据集
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


# 读取数据
def read_tex_graph(file_name):
    if file_name == "twitter":
        path = r"./data/IJCAI19_network_dataset/fb-tt1-copy.edges"
    elif file_name =="facebook":
        path = r"./data/IJCAI19_network_dataset/fb-tt2-copy.edges"
    elif file_name == "dblp":
        path = r'./data/IJCAI19_network_dataset/DBLP.edges'
    elif file_name == "email":
        path = r'./data/IJCAI19_network_dataset/email.ungraph'
    else:
        raise ValueError

    g = nx.Graph()
    with open(path, 'r') as f:
        for line in f.readlines():
            if line:
                if file_name == "email":
                    edge = line.strip().split(' ')
                    # edge = (str(edge[0]), str(edge[1]))
                    edge = (int(edge[0])-1, int(edge[1])-1)
                else:
                    edge = line.strip().split(',')
                    edge = (int(edge[0]), int(edge[1]))
                g.add_edge(edge[0], edge[1])
    if file_name == "dblp":
        attr_path = r'.\data\IJCAI19_network_dataset\DBLPattr.csv'
        attr = np.loadtxt(attr_path,delimiter=',',usecols=[i for i in range(9) if i!=0])

        order_g = nx.Graph()
        nodes = sorted(list(g.nodes()))
        order_dict = dict(zip(nodes, range(nx.number_of_edges(g))))
        for edge in g.edges():
            order_g.add_edge(order_dict[edge[0]], order_dict[edge[1]])
        return order_g, attr


    return g


# 提取结构信息
def structing(layers, G1, G2, G1_degree_dict, G2_degree_dict, attribute, alpha, c,c_num):
    # G1_nodes = list(G1.nodes())
    # G2_nodes = list(G2.nodes())
    G1_nodes = [i for i in range(len(G1))]
    G2_nodes = [i for i in range(len(G2))]
    k1 = k2 = 1
    pp_dist_matrix = {}
    pp_dist_df = pd.DataFrame(np.zeros((G1.number_of_nodes(), G2.number_of_nodes())),
                              index=G1_nodes, columns=G2_nodes)
    # max
    for layer in range(layers + 1):
        L1 = [np.log(k1 * np.max(G1_degree_dict[layer][x]) + np.e) for x in G1_nodes]
        L2 = [np.log(k2 * np.max(G2_degree_dict[layer][x]) + np.e) for x in G2_nodes]
        pp_dist_matrix[layer, 0] = pd.DataFrame(
            np.transpose(np.array(L1 * G2.number_of_nodes()).reshape(-1, G1.number_of_nodes())),
            index=G1_nodes, columns=G2_nodes)
        pp_dist_matrix[layer, 1] = pd.DataFrame(
            np.array(list(L2 * G1.number_of_nodes())).reshape(-1, G2.number_of_nodes()),
            index=G1_nodes, columns=G2_nodes)
        pp_dist_df += abs(pp_dist_matrix[layer, 0] - pp_dist_matrix[layer, 1])

    # min
    for layer in range(layers + 1):
        L1 = [np.log(k1 * np.min(G1_degree_dict[layer][x]) + 1) for x in G1_nodes]
        L2 = [np.log(k2 * np.min(G2_degree_dict[layer][x]) + 1) for x in G2_nodes]
        pp_dist_matrix[layer, 0] = pd.DataFrame(
            np.transpose(np.array(L1 * G2.number_of_nodes()).reshape(-1, G1.number_of_nodes())),
            index=G1_nodes, columns=G2_nodes)
        pp_dist_matrix[layer, 1] = pd.DataFrame(
            np.array(list(L2 * G1.number_of_nodes())).reshape(-1, G2.number_of_nodes()),
            index=G1_nodes, columns=G2_nodes)
        pp_dist_df += abs(pp_dist_matrix[layer, 0] - pp_dist_matrix[layer, 1])

    # medium
    for layer in range(layers + 1):
        L1 = [np.log(k1 * np.percentile(G1_degree_dict[layer][x],50) + 1) for x in G1_nodes]
        L2 = [np.log(k2 * np.percentile(G2_degree_dict[layer][x],50) + 1) for x in G2_nodes]
        pp_dist_matrix[layer, 0] = pd.DataFrame(
            np.transpose(np.array(L1 * G2.number_of_nodes()).reshape(-1, G1.number_of_nodes())),
            index=G1_nodes, columns=G2_nodes)
        pp_dist_matrix[layer, 1] = pd.DataFrame(
            np.array(list(L2 * G1.number_of_nodes())).reshape(-1, G2.number_of_nodes()),
            index=G1_nodes, columns=G2_nodes)
        pp_dist_df += abs(pp_dist_matrix[layer, 0] - pp_dist_matrix[layer, 1])
    #
    for layer in range(layers + 1):
        L1 = [np.log(k1 * np.percentile(G1_degree_dict[layer][x], 25) + 1) for x in G1_nodes]
        L2 = [np.log(k2 * np.percentile(G2_degree_dict[layer][x], 25) + 1) for x in G2_nodes]
        pp_dist_matrix[layer, 0] = pd.DataFrame(
            np.transpose(np.array(L1 * G2.number_of_nodes()).reshape(-1, G1.number_of_nodes())),
            index=G1_nodes, columns=G2_nodes)
        pp_dist_matrix[layer, 1] = pd.DataFrame(
            np.array(list(L2 * G1.number_of_nodes())).reshape(-1, G2.number_of_nodes()),
            index=G1_nodes, columns=G2_nodes)
        pp_dist_df += abs(pp_dist_matrix[layer, 0] - pp_dist_matrix[layer, 1])

    for layer in range(layers + 1):
        L1 = [np.log(k1 * np.percentile(G1_degree_dict[layer][x], 75) + 1) for x in G1_nodes]
        L2 = [np.log(k2 * np.percentile(G2_degree_dict[layer][x], 75) + 1) for x in G2_nodes]
        pp_dist_matrix[layer, 0] = pd.DataFrame(
            np.transpose(np.array(L1 * G2.number_of_nodes()).reshape(-1, G1.number_of_nodes())),
            index=G1_nodes, columns=G2_nodes)
        pp_dist_matrix[layer, 1] = pd.DataFrame(
            np.array(list(L2 * G1.number_of_nodes())).reshape(-1, G2.number_of_nodes()),
            index=G1_nodes, columns=G2_nodes)
        pp_dist_df += abs(pp_dist_matrix[layer, 0] - pp_dist_matrix[layer, 1])




    # pp_dist_df /= 2 ## 需要注释吗

    pp_dist_df = np.exp(-alpha * pp_dist_df)


    if attribute is not None:
        attribute = pd.DataFrame(attribute, index=G1_nodes, columns=G2_nodes)
        pp_dist_df = c * pp_dist_df + attribute * (1 - c)
    struc_neighbor1 = {}
    struc_neighbor2 = {}
    struc_neighbor_sim1 = {}
    struc_neighbor_sim2 = {}

    struc_neighbor_sim1_score = {}
    struc_neighbor_sim2_score = {}
    for i in range(G1.number_of_nodes()):
        pp = pp_dist_df.iloc[i, np.argsort(-pp_dist_df.iloc[i, :])]
        struc_neighbor1[G1_nodes[i]] = list(pp.index[:c_num])
        struc_neighbor_sim1[G1_nodes[i]] = np.array(pp[:c_num])

        struc_neighbor_sim1_score[G1_nodes[i]] = struc_neighbor_sim1[G1_nodes[i]]
        struc_neighbor_sim1[G1_nodes[i]] /= np.sum(struc_neighbor_sim1[G1_nodes[i]])
    pp_dist_df = pp_dist_df.transpose()
    for i in range(G2.number_of_nodes()):
        pp = pp_dist_df.iloc[i, np.argsort(-pp_dist_df.iloc[i, :])]
        struc_neighbor2[G2_nodes[i]] = list(pp.index[:c_num])
        struc_neighbor_sim2[G2_nodes[i]] = np.array(pp[:c_num])

        struc_neighbor_sim2_score[G2_nodes[i]] = struc_neighbor_sim2[G2_nodes[i]]
        struc_neighbor_sim2[G2_nodes[i]] /= np.sum(struc_neighbor_sim2[G2_nodes[i]])

    return struc_neighbor1, struc_neighbor2, struc_neighbor_sim1, struc_neighbor_sim2,pp_dist_df.values


# sinhorn
def doubly_stochastic(P, tau, it,A1,A2):
    """Uses logsumexp for numerical stability."""

    A = P / tau
    for i in range(it):

        # A = A + 1e-4
        A = A - A.logsumexp(dim=1, keepdim=True)
        A = A - A.logsumexp(dim=0, keepdim=True)

    return torch.exp(A)



def node_to_degree(G_degree, SET):
    SET = list(SET)
    SET = sorted([G_degree[x] for x in SET])
    return SET


# degree feature extract
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


def get_sub_graphs(g, node_num,feat):
    N = 0
    layer = 0

    center_node = random.choice(list(range(len(g))))

    while N < node_num:

        layer += 1
        new_g = nx.generators.ego.ego_graph(g,center_node,layer)
        N = len(new_g)

    g = nx.Graph(new_g)
    g1 = nx.Graph()
    ans_dict = {}
    if feat is not None:
        feat_list = []
        cnt = 0
        for node in g.nodes():
            feat_list.append(feat[node,:])
            ans_dict[node] = cnt
            cnt += 1
        feat = np.array(feat_list).reshape(-1, feat[0].shape[-1])
        edges = [(ans_dict[edge[0]], ans_dict[edge[1]]) for edge in g.edges()]
        g1.add_edges_from(edges)
        return g1, feat
    else:
        cnt = 0
        for node in g.nodes():
            ans_dict[node] = cnt
            cnt += 1

    edges = [(ans_dict[edge[0]], ans_dict[edge[1]]) for edge in g.edges()]
    g1.add_edges_from(edges)
    return g1

def get_different_number_graphs(g,feat,remove_num=130):
    new_g = copy.deepcopy(g)
    r_nodes = []

    while len(r_nodes) < remove_num:
        flag = True
        node = random.choice(list(new_g.nodes()))
        for neighbor in new_g.neighbors(node):
            if new_g.degree[neighbor] <= 1:
                flag = False
                break
        if flag:
            new_g.remove_node(node)
            r_nodes.append(node)



    nodes = sorted(list(new_g.nodes()))


    index = [i for i in range(len(nodes))]
    ans_dict = dict(zip(index,nodes))

    r_ans_dict = {ans_dict[i]:i for i in ans_dict.keys()}

    edges = [(r_ans_dict[edge[0]], r_ans_dict[edge[1]]) for edge in new_g.edges()]
    g2 = nx.Graph()
    g2.add_nodes_from(index)
    g2.add_edges_from(edges)

    if feat is not None:
        feat_list = []
        for i in index:
            feat_list.append(feat[ans_dict[i],:])
        feat2 = np.array(feat_list).reshape(-1, feat.shape[-1])
        return g2,ans_dict,feat2
    return g2, ans_dict




def get_k_hop_nodes(g, node, max_hops):
    node_neighbor_dict = {0: [node], 1: list(g.neighbors(node))}
    visited_nodes = [node] + list(g.neighbors(node))
    hop = 1
    while hop < max_hops:
        tmp = []
        for _node in node_neighbor_dict[hop]:
            tmp.extend([i for i in g.neighbors(_node) if i not in visited_nodes])
        hop += 1
        node_neighbor_dict[hop] = list(set(tmp))
        visited_nodes.extend(node_neighbor_dict[hop])

    return node_neighbor_dict
def get_degree(g, node):
    return g.degree[node]

def get_graph_degree_feature(g, max_hops, factor=1):
    feat_matrix = np.zeros((nx.number_of_nodes(g), max_hops * 2 + 2))
    degree_hat_dict = {j: math.log(get_degree(g, j)*factor + 1) for j in g.nodes()}
    for node in tqdm.tqdm(list(g.nodes())):
        node_neighbors_dict = get_k_hop_nodes(g, node, max_hops=max_hops)
        for index, key in enumerate(node_neighbors_dict.keys()):
            if len(node_neighbors_dict[key]) > 0:
                feat_matrix[node, index * 2] = min(
                    [degree_hat_dict[i] for i in node_neighbors_dict[key]])
                feat_matrix[node, index * 2 + 1] = max(
                    [degree_hat_dict[i] for i in node_neighbors_dict[key]])
    return feat_matrix


def jaccad(item):
    fenzi = len(set([i for i in item[0] if i in item[1]]))
    fenmu = len(set(item[1]+item[0])) + 1e-12
    return fenzi/fenmu


def jc_sim(seed_dict, c_g, o_g):
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
    # jc_score[jc_score==0] = 1e-3
    return jc_score


def compute_average_mnc(sim,ans_dict,g1,g2,max_mnc=True):
    pred_dict = greedy_match(sim)
    mnc_matrix = jc_sim(pred_dict,g1,g2)
    node_pairs = [(node, ans_dict[node]) for node in ans_dict.keys()]
    mnc_socres = [mnc_matrix[pair[0],pair[1]] for pair in node_pairs]
    pred_average_score = sum(mnc_socres)/len(mnc_socres)

    if max_mnc:
        mnc_matrix = jc_sim(ans_dict, g1, g2)
        node_pairs = [(node, ans_dict[node]) for node in ans_dict.keys()]
        mnc_socres = [mnc_matrix[pair[0], pair[1]] for pair in node_pairs]
        average_score = sum(mnc_socres) / len(mnc_socres)
        return pred_average_score, average_score

    return pred_average_score


def compute_structural_similarity(layers, G1, G2, attribute, alpha=5, c=0.5):


    G1_degree_dict = cal_degree_dict(list(G1.nodes()), G1, layers)
    G2_degree_dict = cal_degree_dict(list(G2.nodes()), G2, layers)

    G1_nodes = [i for i in range(len(G1))]
    G2_nodes = [i for i in range(len(G2))]
    k1 = k2 = 1
    pp_dist_matrix = {}
    pp_dist_df = pd.DataFrame(np.zeros((G1.number_of_nodes(), G2.number_of_nodes())),
                              index=G1_nodes, columns=G2_nodes)
    # max
    for layer in range(layers + 1):
        L1 = [np.log(k1 * np.max(G1_degree_dict[layer][x]) + np.e) for x in G1_nodes]
        L2 = [np.log(k2 * np.max(G2_degree_dict[layer][x]) + np.e) for x in G2_nodes]
        pp_dist_matrix[layer, 0] = pd.DataFrame(
            np.transpose(np.array(L1 * G2.number_of_nodes()).reshape(-1, G1.number_of_nodes())),
            index=G1_nodes, columns=G2_nodes)
        pp_dist_matrix[layer, 1] = pd.DataFrame(
            np.array(list(L2 * G1.number_of_nodes())).reshape(-1, G2.number_of_nodes()),
            index=G1_nodes, columns=G2_nodes)
        pp_dist_df += abs(pp_dist_matrix[layer, 0] - pp_dist_matrix[layer, 1])

    # min
    for layer in range(layers + 1):
        L1 = [np.log(k1 * np.min(G1_degree_dict[layer][x]) + 1) for x in G1_nodes]
        L2 = [np.log(k2 * np.min(G2_degree_dict[layer][x]) + 1) for x in G2_nodes]
        pp_dist_matrix[layer, 0] = pd.DataFrame(
            np.transpose(np.array(L1 * G2.number_of_nodes()).reshape(-1, G1.number_of_nodes())),
            index=G1_nodes, columns=G2_nodes)
        pp_dist_matrix[layer, 1] = pd.DataFrame(
            np.array(list(L2 * G1.number_of_nodes())).reshape(-1, G2.number_of_nodes()),
            index=G1_nodes, columns=G2_nodes)
        pp_dist_df += abs(pp_dist_matrix[layer, 0] - pp_dist_matrix[layer, 1])

    # medium
    for layer in range(layers + 1):
        L1 = [np.log(k1 * np.percentile(G1_degree_dict[layer][x],50) + 1) for x in G1_nodes]
        L2 = [np.log(k2 * np.percentile(G2_degree_dict[layer][x],50) + 1) for x in G2_nodes]
        pp_dist_matrix[layer, 0] = pd.DataFrame(
            np.transpose(np.array(L1 * G2.number_of_nodes()).reshape(-1, G1.number_of_nodes())),
            index=G1_nodes, columns=G2_nodes)
        pp_dist_matrix[layer, 1] = pd.DataFrame(
            np.array(list(L2 * G1.number_of_nodes())).reshape(-1, G2.number_of_nodes()),
            index=G1_nodes, columns=G2_nodes)
        pp_dist_df += abs(pp_dist_matrix[layer, 0] - pp_dist_matrix[layer, 1])
    #
    for layer in range(layers + 1):
        L1 = [np.log(k1 * np.percentile(G1_degree_dict[layer][x], 25) + 1) for x in G1_nodes]
        L2 = [np.log(k2 * np.percentile(G2_degree_dict[layer][x], 25) + 1) for x in G2_nodes]
        pp_dist_matrix[layer, 0] = pd.DataFrame(
            np.transpose(np.array(L1 * G2.number_of_nodes()).reshape(-1, G1.number_of_nodes())),
            index=G1_nodes, columns=G2_nodes)
        pp_dist_matrix[layer, 1] = pd.DataFrame(
            np.array(list(L2 * G1.number_of_nodes())).reshape(-1, G2.number_of_nodes()),
            index=G1_nodes, columns=G2_nodes)
        pp_dist_df += abs(pp_dist_matrix[layer, 0] - pp_dist_matrix[layer, 1])

    for layer in range(layers + 1):
        L1 = [np.log(k1 * np.percentile(G1_degree_dict[layer][x], 75) + 1) for x in G1_nodes]
        L2 = [np.log(k2 * np.percentile(G2_degree_dict[layer][x], 75) + 1) for x in G2_nodes]
        pp_dist_matrix[layer, 0] = pd.DataFrame(
            np.transpose(np.array(L1 * G2.number_of_nodes()).reshape(-1, G1.number_of_nodes())),
            index=G1_nodes, columns=G2_nodes)
        pp_dist_matrix[layer, 1] = pd.DataFrame(
            np.array(list(L2 * G1.number_of_nodes())).reshape(-1, G2.number_of_nodes()),
            index=G1_nodes, columns=G2_nodes)
        pp_dist_df += abs(pp_dist_matrix[layer, 0] - pp_dist_matrix[layer, 1])




    # pp_dist_df /= 2 ## 需要注释吗

    pp_dist_df = np.exp(-alpha * pp_dist_df)


    if attribute is not None:
        attribute = pd.DataFrame(attribute, index=G1_nodes, columns=G2_nodes)
        pp_dist_df = c * pp_dist_df + attribute * (1 - c)
    degree_similarity = np.array(pp_dist_df)
    return degree_similarity

import torch.nn.functional as F
def get_optimal_loss(emb11, emb12,beta=0.5,out_iters=10,inner_iters=1):
    n = emb11.shape[0]
    m = emb12.shape[0]

    sigma = torch.ones(size=(m,1)).double()
    T = torch.ones(size=(n,1)) @ torch.ones(size=(1,m))
    T = T.double()
    # T = F.normalize(emb11, p=2, dim=1) @ F.normalize(emb12, p=2, dim=1).T

    # 求取cost function
    C = mutual_cost_mat(emb11,emb12,cost_type="cosine")

    # C =  1.0 - F.normalize(emb11, p=2, dim=1) @ F.normalize(emb12, p=2, dim=1).T


    # min_score = C.min()
    # max_score = C.max()
    # threshold = min_score + beta * (max_score - min_score)
    # C = torch.nn.functional.relu(C - threshold)

    A = torch.exp(-C/beta) # (n,m)


    for out_iter in range(out_iters):

        Q = A * T #(n,m)
        for inner_iter in range(inner_iters):
            delta = 1.0/(n * Q @ sigma) # (n,1)
            sigma = 1.0/(m * Q.T @ delta) # (m,1)
            T = torch.diag(delta.reshape(-1,)) @ Q @ torch.diag(sigma.reshape(-1))

    loss = torch.sum(C * T)


    return loss,T



def mutual_cost_mat(embs1,embs2,cost_type = 'cosine'):

    if cost_type == 'cosine':
        # cosine similarity
        energy1 = torch.sqrt(torch.sum(embs1 ** 2, dim=1, keepdim=True))  # (batch_size1, 1)
        energy2 = torch.sqrt(torch.sum(embs2 ** 2, dim=1, keepdim=True))  # (batch_size2, 1)
        cost = 1-torch.exp(-(1-torch.matmul(embs1, torch.t(embs2))/(torch.matmul(energy1, torch.t(energy2))+1e-5)))
    else:
        # Euclidean distance
        embs = torch.matmul(embs1, torch.t(embs2))  # (batch_size1, batch_size2)
        # (batch_size1, batch_size2)
        embs_diag1 = torch.diag(torch.matmul(embs1, torch.t(embs1))).view(-1, 1).repeat(1, embs2.size(0))
        # (batch_size2, batch_size1)
        embs_diag2 = torch.diag(torch.matmul(embs2, torch.t(embs2))).view(-1, 1).repeat(1, embs1.size(0))
        cost = 1-torch.exp(-(embs_diag1 + torch.t(embs_diag2) - 2 * embs)/embs1.size(1))
    return cost


# def compute_edge_overlap_rate(adj1,adj2,ans_dict):
#     n1 = adj1.shape[0]
#     n2 = adj2.shape[0]
#     P = np.zeros((n1,n2))
#     nodes = list(ans_dict.keys())
#     r_nodes = [ans_dict[i] for i in nodes]
#     P[nodes,r_nodes] = 1
#     adj2 = P.T @ adj2 @ P
#     num1 = np.sum(adj1)
#     num2 = np.sum(adj2)
#     tmp = (adj1+adj2)/2.0
#     tmp[tmp<1] = 0
#     common = np.sum(tmp)
#     rate = common/(num1+num2-common)
#     return rate


def compute_edge_overlap_rate(g1,g2,ans_dict):
    cnt = 0
    for edge in g1.edges():
        if ans_dict[edge[1]] in g2.neighbors(ans_dict[edge[0]]):
            cnt += 1
    e1 = nx.number_of_edges(g1)
    e2 = nx.number_of_edges(g2)
    rate = cnt/(e1+e2-cnt)
    return rate



def batch_generator(data, batch_size, shuffle=True):
    if shuffle:
        data = [data[i] for i in np.random.permutation(len(data))]
    batch_count = 0
    while True:
        if batch_count * batch_size +batch_size>= len(data):
            batch_count = 0
            if shuffle:
                data = [data[i] for i in np.random.permutation(len(data))]
        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        yield data[start:end]


def print_run_time(func):
    def wrapper(*args, **kw):
        local_time = time.time()
        func(*args, **kw)
        print ('current Function [%s] run time is %.2f' % (func.__name__ ,time.time() - local_time))
    return wrapper

