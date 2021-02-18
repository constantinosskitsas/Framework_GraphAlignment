from algorithms import regal, eigenalign, conealign, netalign
from data import ReadFile
from evaluation import evaluation, evaluation_design
from sacred import Experiment


ex = Experiment("experiment")


@ex.config
def global_config():

    gt = "data/noise_level_2/gt_4.txt"
    data1 = "data/noise_level_2/edges_4.txt"
    data2 = "data/noise_level_1/arenas_orig.txt"

    # gt = "data/test/gt.txt"
    # data1 = "data/test/edges.txt"
    # data2 = "data/test/arenas_orig.txt"

    # data1 = "data/test/A.txt"
    # data2 = "data/test/B.txt"

    G2 = ReadFile.edgelist_to_adjmatrix1(data2)
    G1 = ReadFile.edgelist_to_adjmatrix1(data1)
    G3 = ReadFile.edgelist_to_adjmatrix1(data2)
    # G3=evaluation_design.remove_edges_directed(G3)

    adj = ReadFile.edgelist_to_adjmatrixR(data1, data2)
    adj1 = ReadFile.Edge_Removed_edgelist_to_adjmatrixR(G2, G3)
    gma, gmb = ReadFile.gt1(gt)


@ex.capture
def eval_regal(_log, gma, gmb, adj, adj1):
    alignmatrix = regal.main(adj)
    mar, mbr = evaluation.transformRAtoNormalALign(alignmatrix)
    acc1 = evaluation.accuracy(gma, gmb, mbr, mar)
    _log.info(f"acc1: {acc1}")

    alignmatrix = regal.main(adj1)
    mar, mbr = evaluation.transformRAtoNormalALign(alignmatrix)
    acc = evaluation.check_with_identity(mbr+1)
    _log.info(f"acc: {acc}")

    return acc, acc1


@ex.capture
def eval_eigenalign(_log, gma, gmb, G1, G2, G3):
    ma1, mb1, _, _ = eigenalign.main(G1, G2, 8, "lowrank_svd_union", 3)
    ma2, mb2, _, _ = eigenalign.main(G2, G3, 8, "lowrank_svd_union", 3)

    acc = evaluation.accuracy(gma+1, gmb+1, mb1, ma1)
    _log.info(f"acc: {acc}")

    acc1 = evaluation.check_with_identity(mb2)
    _log.info(f"acc1: {acc1}")

    return acc, acc1


@ex.capture
def eval_conealign(_log, gma, gmb, adj, adj1):
    alignmatrix = conealign.main(adj)
    mar, mbr = evaluation.transformRAtoNormalALign(alignmatrix)
    acc1 = evaluation.accuracy(gma, gmb, mbr, mar)
    _log.info(f"acc1: {acc1}")

    alignmatrix = conealign.main(adj1)
    mar, mbr = evaluation.transformRAtoNormalALign(alignmatrix)
    acc = evaluation.check_with_identity(mbr+1)
    _log.info(f"acc: {acc}")

    return acc, acc1


@ex.capture
def eval_netalign(_log):
    import numpy as np

    S = "data/test_/S.txt"
    li = "data/test_/li.txt"
    lj = "data/test_/lj.txt"

    S = ReadFile.edgelist_to_adjmatrix1(S)
    li = np.loadtxt(li)
    lj = np.loadtxt(lj)

    alignmatrix = netalign.main(S, li, lj, 0, 1)

    _log.info(alignmatrix)


@ex.capture
def playground(_log, gma, gmb, G1, G2, G3, adj, adj1):
    import networkx as nx

    # G = nx.Graph()
    # G.add_nodes_from(range(10))
    # G.add_edge(1, 2)
    # print(len(G))
    # print(len(G))
    B = nx.Graph()
    # Add nodes with the node attribute "bipartite"
    B.add_nodes_from([1, 2, 3, 4], bipartite=0)
    B.add_nodes_from(["a", "b", "c"], bipartite=1)
    # Add edges only between nodes of opposite node sets
    B.add_edges_from([(1, "a"), (1, "b"), (2, "b"),
                      (2, "c"), (3, "c"), (4, "a"), (1, 2)])

    print(nx.bipartite.maximum_matching(B))


@ex.automain
def main():
    # eval_regal()
    # eval_eigenalign()
    # eval_conealign()
    eval_netalign()
    # playground()
