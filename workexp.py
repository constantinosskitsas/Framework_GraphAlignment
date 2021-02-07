from algorithms import regal, AlignNetworks_EigenAlign
from data import ReadFile
from evaluation import evaluation
from evaluation import evaluation_design

def main():
    data1 = "data/noise_level_2/edges_4.txt"
    data2 = "data/noise_level_1/arenas_orig.txt"
    gt = "data/noise_level_2/gt_4.txt"
    G2 = ReadFile.edgelist_to_adjmatrix1(data2)
    G1 = ReadFile.edgelist_to_adjmatrix1(data1)
    G3= ReadFile.edgelist_to_adjmatrix1(data2)
    #G3=evaluation_design.remove_edges_directed(G3)
    adj1=ReadFile.Edge_Removed_edgelist_to_adjmatrixR(G2,G3)
    adj = ReadFile.edgelist_to_adjmatrixR(data1, data2)
    gma, gmb = ReadFile.gt1(gt)
   # ma1, mb1, _, _ = AlignNetworks_EigenAlign.align_networks_eigenalign(
    #    G1, G2, 8, "lowrank_svd_union", 3)
    #ma2, mb2, _, _ = AlignNetworks_EigenAlign.align_networks_eigenalign(
       # G2, G3, 8, "lowrank_svd_union", 3)
    lreaacc=0
    lreaacc1 = 0
   # lreaacc = evaluation.accuracy(gma+1, gmb+1, mb1, ma1)
    #print(mb2)
    #lreaacc1 = evaluation.check_with_identity(mb2)
    print(lreaacc)
    print(lreaacc1)
    args1 = regal.parse_args()
    regalacc1 = 0
    regalacc=0
    #for i in range(0, 10):
    alignmatrix = regal.main(adj, args1)
    mar, mbr = evaluation.transformRAtoNormalALign(alignmatrix)
    regalacc1 = evaluation.accuracy(gma, gmb, mbr, mar)
    print(regalacc)
    #regalacc1 = regalacc+regalacc1
    regalacc2 = 0
    # for i in range(0, 10):
    alignmatrix = regal.main(adj1, args1)
    mar, mbr = evaluation.transformRAtoNormalALign(alignmatrix)
    regalacc = evaluation.check_with_identity(mbr+1)
    return regalacc,regalacc1, lreaacc,lreaacc1


if __name__ == "__main__":
    hi = 0
    # for i in range(10):
    #hi = hi + main()
    proto, deftero,trito,tetarto = main()
    print("regal result ", proto, " LREA result ", deftero," s ",trito, " sds", tetarto)
