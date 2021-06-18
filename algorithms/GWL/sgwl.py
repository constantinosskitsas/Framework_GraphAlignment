# from .model.GromovWassersteinLearning import GromovWassersteinLearning
# import torch.optim as optim
# from torch.optim import lr_scheduler
# import torch
import numpy as np
import scipy.sparse as sps
# import scipy.sparse as sps
from .methods import DataIO, GromovWassersteinGraphToolkit as GwGt
import networkx as nx
import torch

# methods = ['gwl', 's-gwl-3', 's-gwl-2', 's-gwl-1']
cluster_num = [2, 4, 8]
partition_level = [3, 2, 1]


def main(data, ot_dict, mn, max_cpu=0):

    Src = data['Src']
    Tar = data['Tar']
    # print(Src.tolist())
    p_s, cost_s, idx2node_s = DataIO.extract_graph_info(
        nx.Graph(Src), weights=None)
    # print(cost_s.A.tolist())
    p_s /= np.sum(p_s)
    p_t, cost_t, idx2node_t = DataIO.extract_graph_info(
        nx.Graph(Tar), weights=None)
    p_t /= np.sum(p_t)

    if max_cpu > 0:
        torch.set_num_threads(max_cpu)

    ot_dict = {
        **ot_dict,
        'outer_iteration': Src.shape[0]
    }

    if mn == 0:
        pairs_idx, pairs_name, pairs_confidence, trans = GwGt.direct_graph_matching(
            cost_s, cost_t, p_s, p_t, idx2node_s, idx2node_t, ot_dict)
    else:
        pairs_idx, pairs_name, pairs_confidence, trans = GwGt.recursive_direct_graph_matching(
            0.5 * (cost_s + cost_s.T), 0.5 * (cost_t + cost_t.T), p_s, p_t,
            idx2node_s, idx2node_t, ot_dict, weights=None, predefine_barycenter=False,
            cluster_num=2, partition_level=3, max_node_num=0
        )

    pairs = np.array(pairs_name)[::-1].T

    # print(pairs)
    # print(pairs_idx)
    # print(pairs.shape)

    # res = sps.csr_matrix((np.ones(pairs.shape[1]), pairs)).A
    # # res[:, -1] = 0.01
    # # print(res)
    # print(trans.shape)
    # print(pairs_name)

    # return res
    return trans

    # Src = data['Src']
    # Tar = data['Tar']

    # # Se = np.array(sps.find(Src)[:2]).T
    # # Te = np.array(sps.find(Tar)[:2]).T
    # Se = np.array(sps.find(sps.csr_matrix(Src))[:2]).T
    # Te = np.array(sps.find(sps.csr_matrix(Tar))[:2]).T

    # data = {
    #     'src_index': {float(i): i for i in range(np.amax(Se) + 1)},
    #     'src_interactions': Se.tolist(),
    #     'tar_index': {float(i): i for i in range(np.amax(Te) + 1)},
    #     'tar_interactions': Te.tolist(),
    #     'mutual_interactions': None
    # }

    # hyperpara_dictt = {
    #     'src_number': len(data['src_index']),
    #     'tar_number': len(data['tar_index']),
    #     **hyperpara_dict
    # }

    # if max_cpu > 0:
    #     torch.set_num_threads(max_cpu)

    # gwd_model = GromovWassersteinLearning(hyperpara_dictt)

    # # initialize optimizer
    # optimizer = optim.Adam(gwd_model.gwl_model.parameters(), lr=lr)

    # scheduler = lr_scheduler.ExponentialLR(
    #     optimizer, gamma=gamma) if gamma else None

    # # Gromov-Wasserstein learning
    # gwd_model.train_without_prior(
    #     data, optimizer, opt_dict, scheduler=scheduler)
    # cost12 = gwd_model.getCostm()
    # # gwd_model.evaluation_recommendation1()
    # return gwd_model.trans, cost12
