import AlignNetworks_EigenAlign
import ReadFile
import regal
from regal import *
from ReadFile import *
from LREA import *
from evaluation_design import *
import evaluation


def main():
    data1="data/noise_level_1/edges_2.txt"
    data2="data/noise_level_1/arenas_orig.txt"
    gt="data/noise_level_1/gt_2.txt"
    G2 = ReadFile.edgelist_to_adjmatrix1(data2)
    G1 = ReadFile.edgelist_to_adjmatrix1(data1)
    adj = edgelist_to_adjmatrixR(data1, data2)
    gma, gmb = ReadFile.gt1(gt)
    ma1,mb1,_,_=AlignNetworks_EigenAlign.align_networks_eigenalign(G1, G2, 8, "lowrank_svd_union", 3)
    lreaacc=evaluation.accuracy(gma+1,gmb+1,mb1,ma1)
    print(lreaacc)
    args1=regal.parse_args()
    alignmatrix=regal.main(adj,args1)
    mar,mbr= evaluation.transformRAtoNormalALign(alignmatrix)
    regalacc = evaluation.accuracy(gma, gmb, mbr, mar)
    print(regalacc)
    return mar

if __name__ == "__main__":
    hi = 0
    #for i in range(10):
        #hi = hi + main()
    print(main())
