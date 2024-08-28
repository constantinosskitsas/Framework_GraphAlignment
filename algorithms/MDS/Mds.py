import numpy as np
import networkx as nx
import torch
from algorithms.FUGAL.pred import feature_extraction,eucledian_dist,convex_init,Degree_Features
import torch
import pandas as pd
import numpy as np
import scipy
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
from algorithms.MDS.joint_mds import JointMDS
from scipy.sparse import csr_matrix
import argparse
import pickle
import warnings
import networkx as nx

def get_quadratic_inverse_weight(shortest_path):
    w = 1.0 / shortest_path**4
    w[np.isinf(w)] = 0.0
    w /= w.sum()
    return w

def compute_shortest_path(adj):
    adj.data = 1.0 / (1.0 + adj.data)
    #adj=1.0/(1.0+adj)
    # adj.data = 1. - adj.data
    adj = dijkstra(csgraph=adj, directed=False, return_predecessors=False)
    adj /= adj.mean()
    return adj

def normalize_adj(adj):
    degree = np.asarray(adj.sum(1))
    d_inv_sqrt = np.power(degree, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    # print("here")
    # print(d_inv_sqrt.shape)
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)  # .tocoo()

def main(data, **args):
    print("Mds")
    A = data['Src']
    B = data['Tar']
    A = csr_matrix(A)
    B = csr_matrix(B)
    adj_s_normalized = normalize_adj(A)
    adj_t_normalized = normalize_adj(B)
    adj_s_normalized = compute_shortest_path(adj_s_normalized)
    adj_t_normalized = compute_shortest_path(adj_t_normalized)
    w1 = get_quadratic_inverse_weight(adj_s_normalized)
    w2 = get_quadratic_inverse_weight(adj_t_normalized)
    w1 = torch.from_numpy(w1)
    w2 = torch.from_numpy(w2)
    torch.manual_seed(1)
    JMDS = JointMDS(
        n_components=args['n_components'],
        alpha=args['alpha'],
        #alpha=0.1,
        max_iter=args['max_iter'],
        eps=args['eps'],
        #eps=1,
        tol=args['tol'],
        min_eps=args['min_eps'],
        eps_annealing=args['eps_annealing'],
        alpha_annealing=args['alpha_annealing'],
        gw_init=args['gw_init'],
        return_stress=args['return_stress']
    )
    Z1, Z2, P = JMDS.fit_transform(
        torch.from_numpy(adj_s_normalized),
        torch.from_numpy(adj_t_normalized),
        w1=w1,
        w2=w2#,
        #a=torch.from_numpy(weight_s),
        #b=torch.from_numpy(weight_t),
    )
    cost_matrix = P.numpy()
    return cost_matrix,cost_matrix