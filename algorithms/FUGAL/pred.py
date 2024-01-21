import numpy as np
import math
import torch
from tqdm.auto import tqdm
import networkx as nx
import time
import matplotlib.pyplot as plt
from multiprocessing import Pool
import scipy
from sklearn.metrics.pairwise import euclidean_distances
from algorithms.FUGAL.sinkhorn import sinkhorn,sinkhorn_epsilon_scaling,sinkhorn_knopp,sinkhorn_stabilized
    
def plot(graph1, graph2):
    plt.figure(figsize=(12,4))
    plt.subplot(121)

    nx.draw(graph1)
    plt.subplot(122)

    nx.draw(graph2)
    plt.savefig('x1.png')

def feature_extraction(G,simple):
    """Node feature extraction.

    Parameters
    ----------

    G (nx.Graph): a networkx graph.

    Returns
    -------

    node_features (float): the Nx7 matrix of node features."""

    # necessary data structures
    node_features = np.zeros(shape=(G.number_of_nodes(), 7))
    node_list = sorted(G.nodes())
    node_degree_dict = dict(G.degree())
    node_clustering_dict = dict(nx.clustering(G))
    egonets = {n: nx.ego_graph(G, n) for n in node_list}

    # node degrees
    degs = [node_degree_dict[n] for n in node_list]

    # clustering coefficient
    clusts = [node_clustering_dict[n] for n in node_list]

    # average degree of neighborhood
    neighbor_degs = [
        np.mean([node_degree_dict[m] for m in egonets[n].nodes if m != n])
        if node_degree_dict[n] > 0
        else 0
        for n in node_list
    ]

    # average clustering coefficient of neighborhood
    neighbor_clusts = [
        np.mean([node_clustering_dict[m] for m in egonets[n].nodes if m != n])
        if node_degree_dict[n] > 0
        else 0
        for n in node_list
    ]

    # number of edges in the neighborhood
    neighbor_edges = [
        egonets[n].number_of_edges() if node_degree_dict[n] > 0 else 0
        for n in node_list
    ]

    # number of outgoing edges from the neighborhood
    # the sum of neighborhood degrees = 2*(internal edges) + external edges
    # node_features[:,5] = node_features[:,0] * node_features[:,2] - 2*node_features[:,4]
    neighbor_outgoing_edges = [
        len(
            [
                edge
                for edge in set.union(*[set(G.edges(j)) for j in egonets[i].nodes])
                if not egonets[i].has_edge(*edge)
            ]
        )
        for i in node_list
    ]

    # number of neighbors of neighbors (not in neighborhood)
    neighbors_of_neighbors = [
        len(
            set([p for m in G.neighbors(n) for p in G.neighbors(m)])
            - set(G.neighbors(n))
            - set([n])
        )
        if node_degree_dict[n] > 0
        else 0
        for n in node_list
    ]

    # assembling the features
    node_features[:, 0] = degs
    node_features[:, 1] = clusts
    node_features[:, 2] = neighbor_degs
    node_features[:, 3] = neighbor_clusts
    if (simple==False):
        node_features[:, 4] = neighbor_edges #create if statement
        node_features[:, 5] = neighbor_outgoing_edges#
        node_features[:, 6] = neighbors_of_neighbors#

    node_features = np.nan_to_num(node_features)
    return np.nan_to_num(node_features)

def eucledian_dist(F1, F2, n):
    D = euclidean_distances(F1, F2)
    return D

def dist(A, B, P):
    obj = np.linalg.norm(np.dot(A, P) - np.dot(P, B))
    return obj*obj/2

def convex_init(A, B, D, mu, niter):
    n = len(A)
    P = torch.eye(n, dtype = torch.float64)
    ones = torch.ones(n, dtype = torch.float64)
    mat_ones = torch.ones((n, n), dtype = torch.float64)
    reg = 1.0

    for i in range(niter):
        for it in range(1, 11):
            G = (torch.mm(torch.mm(A.T, A), P) - torch.mm(torch.mm(A.T, P), B) - torch.mm(torch.mm(A, P), B.T) + torch.mm(torch.mm(P, B), B.T))/2 + mu*D + i*(mat_ones - 2*P)
            q = sinkhorn(ones, ones, G, reg, maxIter = 500, stopThr = 1e-3)
            alpha = 2.0 / float(2.0 + it)
            P = P + alpha * (q - P)
    return P

def convex_init1(A, B, D, mu, niter):
    n = len(A)
    P = torch.eye(n, dtype = torch.float64)
    ones = torch.ones(n, dtype = torch.float64)
    mat_ones = torch.ones((n, n), dtype = torch.float64)
    reg = 1.0
    K=mu*D 
    AA=torch.mm(A.T, A)
    BT=torch.mm(B.T, A)
    BB=torch.mm(B,B.T)
    AB=torch.mm(A.T, B)
    for i in range(niter):
        for it in range(1, 11):
            #G = (torch.mm(AA, P) - torch.mm(torch.mm(A.T, P), B) - torch.mm(torch.mm(A, P), B.T) + torch.mm(torch.mm(P, B), B.T))/2 + K + i*(mat_ones - 2*P)
            #G = (torch.mm(AA, P) - torch.mm(torch.mm(A.T, P), B) - torch.mm(torch.mm(A,P ), B.T) + torch.mm(P,BB))/2 + K + i*(mat_ones - 2*P)
            #G = (torch.mm(AA, P) - torch.mm(torch.mm(A.T,P), B) - torch.mm(torch.mm(A,P ), B.T) + torch.mm(P,BB))/2 + K + i*(mat_ones - 2*P)
            #G=2*torch.mm(AA, P)-torch.mm(BT, P)-torch.mm(AB, P)+torch.mm(torch.mm(B.T, P), B)-torch.mm(torch.mm(A.T, P), B)-torch.mm(torch.mm(A, P), B.T)+torch.mm(torch.mm(B, P), B.T)+ K + i*(mat_ones - 2*P)
            G=-torch.mm(torch.mm(A.T, P), B)-torch.mm(torch.mm(A, P), B.T)+ K + i*(mat_ones - 2*P)
            q = sinkhorn(ones, ones, G, reg, maxIter = 500, stopThr = 1e-3)
            alpha = 2.0 / float(2.0 + it)
            P = P + alpha * (q - P)
    return P
def convertToPermHungarian(M, n1, n2):
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(M, maximize=True)
    n = len(M)

    P = np.zeros((n, n))
    ans = []
    for i in range(n):
        P[row_ind[i]][col_ind[i]] = 1
        if (row_ind[i] >= n1) or (col_ind[i] >= n2):
            continue
        ans.append((row_ind[i], col_ind[i]))
    return P, ans

def convertToPermGreedy(M, n1, n2):
    n = len(M)
    indices = torch.argsort(M.flatten())
    row_done = np.zeros(n)
    col_done = np.zeros(n)

    P = np.zeros((n, n))
    ans = []
    for i in range(n*n):
        cur_row = int(indices[n*n - 1 - i]/n)
        cur_col = int(indices[n*n - 1 - i]%n)
        if (row_done[cur_row] == 0) and (col_done[cur_col] == 0):
            P[cur_row][cur_col] = 1
            row_done[cur_row] = 1
            col_done[cur_col] = 1
            if (cur_row >= n1) or (cur_col >= n2):
                continue
            ans.append((cur_row, cur_col))
    return P, ans

def convertToPerm(A, B, M, n1, n2):
    P_hung, ans_hung = convertToPermHungarian(M, n1, n2)
    P_greedy, ans_greedy = convertToPermGreedy(M, n1, n2)
    dist_hung = dist(A, B, P_hung)
    dist_greedy = dist(A, B, P_greedy)
    if dist_hung < dist_greedy:
        return P_hung, ans_hung
    else:
        return P_greedy, ans_greedy

def align_new(Gq, Gt, mu, niter):
    n1 = len(Gq.nodes())
    n2 = len(Gt.nodes())
    n = max(n1, n2)
    for i in range(n1, n):
        Gq.add_node(i)
    for i in range(n2, n):
        Gt.add_node(i)

    A = torch.tensor(nx.to_numpy_array(Gq), dtype = torch.float64)
    B = torch.tensor(nx.to_numpy_array(Gt), dtype = torch.float64)
    F1 = feature_extraction(Gq)
    F2 = feature_extraction(Gt)
    D = eucledian_dist(F1, F2, n)
    D = torch.tensor(D, dtype = torch.float64)

    P = convex_init(A, B, D, mu, niter)
    P_perm, ans = convertToPermHungarian(P, n1, n2)
    return ans

def predict_alignment(queries, targets, mu = 2, niter = 15):
    n = len(queries)
    mapping = []
    times = []
    for i in tqdm(range(n)):
        t1 = time.time()
        ans = align_new(queries[i], targets[i], mu, niter)
        mapping.append(ans)
        t2 = time.time()
        times.append(t2 - t1)
    return mapping, times