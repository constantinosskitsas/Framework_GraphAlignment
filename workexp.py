from algorithms import regal, eigenalign, conealign
from data import ReadFile
from evaluation import evaluation, evaluation_design
from sacred import Experiment

ex = Experiment("experiment")


@ex.config
def global_config():
    gt = "data/noise_level_2/gt_4.txt"
    data1 = "data/noise_level_2/edges_4.txt"
    data2 = "data/noise_level_1/arenas_orig.txt"

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
def playground(_log, gma, gmb, G1, G2, G3, adj, adj1):
    pass


@ex.automain
def main():
    # eval_regal()
    # eval_eigenalign()
    # eval_conealign()
    # playground()
