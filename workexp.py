from algorithms import regal, AlignNetworks_EigenAlign
from data import ReadFile
from evaluation import evaluation


def main():
    data1 = "data/noise_level_2/edges_4.txt"
    data2 = "data/noise_level_1/arenas_orig.txt"
    gt = "data/noise_level_2/gt_4.txt"
    G2 = ReadFile.edgelist_to_adjmatrix1(data2)
    G1 = ReadFile.edgelist_to_adjmatrix1(data1)
    adj = ReadFile.edgelist_to_adjmatrixR(data1, data2)
    gma, gmb = ReadFile.gt1(gt)
    ma1, mb1, _, _ = AlignNetworks_EigenAlign.align_networks_eigenalign(
        G1, G2, 8, "lowrank_svd_union", 3)
    lreaacc = evaluation.accuracy(gma+1, gmb+1, mb1, ma1)
    print(lreaacc)
    args1 = regal.parse_args()
    regalacc1 = 0
    for i in range(0, 10):
        alignmatrix = regal.main(adj, args1)
        mar, mbr = evaluation.transformRAtoNormalALign(alignmatrix)
        regalacc = evaluation.accuracy(gma, gmb, mbr, mar)
        print(regalacc)
        regalacc1 = regalacc+regalacc1

    return regalacc1/10, lreaacc


if __name__ == "__main__":
    hi = 0
    # for i in range(10):
    #hi = hi + main()
    proto, deftero = main()
    print("regal result ", proto, " LREA result ", deftero)
