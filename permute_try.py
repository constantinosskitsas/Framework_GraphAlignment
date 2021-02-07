import networkx as nx
import numpy as np
from data import ReadFile
from random import randrange
import random
from algorithms import regal, AlignNetworks_EigenAlign
from evaluation import evaluation
def create_mapping(n):
    d = dict()
    mylist = list(range(n))
    for _ in range(250):
        random_item_from_list = random.choice(mylist)
        mylist.remove(random_item_from_list)
        random_item_from_list1 = random.choice(mylist)
        mylist.remove(random_item_from_list1)
        d[random_item_from_list]=random_item_from_list1
        d[random_item_from_list1]=random_item_from_list
    return d


def main():
    #G1 = ReadFile.edgelist_to_adjmatrix1("magkas.txt")
    data2 = "data/noise_level_1/arenas_orig.txt"
    G1 = ReadFile.edgelist_to_adjmatrix1(data2)
    G2 = nx.Graph(ReadFile.edgelist_to_adjmatrix1(data2))
    mapp=create_mapping(len(G2))
    B = nx.relabel_nodes(G2, mapp, copy=True)
    #G2 = nx.to_numpy_array(B)
    nx.write_edgelist(B, "test.edgelist", data=False)
    G3= ReadFile.edgelist_to_adjmatrix1("test.edgelist")
    gmb=np.zeros(len(G1))
    for i in range(len(gmb)):
        gmb[i]=-1
    for i in mapp:
        print(mapp[i])
        gmb[i]=mapp[i]

    for i in range (len(G1)):
        if (gmb[i]==-1):
            gmb[i]=i
    gma=range(len(gmb))
    args1 = regal.parse_args()

    adj = ReadFile.Edge_Removed_edgelist_to_adjmatrixR(G1, G3)
    alignmatrix = regal.main(adj, args1)
    mar, mbr = evaluation.transformRAtoNormalALign(alignmatrix)
    reg = evaluation.accuracy2(gmb,mbr)
    print(mbr)
    print(gmb)
    print(reg)
if __name__ == "__main__":
    hi = 0
    main()
