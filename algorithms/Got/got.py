import numpy as np
import networkx as nx
import torch
from algorithms.FUGAL.pred import feature_extraction,eucledian_dist,convex_init,Degree_Features

def main(data, **args):
    print("got")
    Src = data['Src']
    Tar = data['Tar']
    for i in range(Src.shape[0]):
        row_sum1 = np.sum(Src[i, :])

    # If the sum of the row is zero, add a self-loop
        if row_sum1 == 0:
            Src[i, i] = 1
    for i in range(Tar.shape[0]):
        row_sum = np.sum(Tar[i, :])

    # If the sum of the row is zero, add a self-loop
        if row_sum == 0:
            Tar[i, i] = 1
    n1 = Tar.shape[0]
    n2 = Src.shape[0]
    n = max(n1, n2)
    torch.set_num_threads(40)
    Src1=nx.from_numpy_array(Src)
    Tar1=nx.from_numpy_array(Tar)
    A = torch.tensor((Src), dtype = torch.float64)
    B = torch.tensor((Tar), dtype = torch.float64)
    simple=True 
    F1 = feature_extraction(Src1,simple)
    F2 = feature_extraction(Tar1,simple)
    D = eucledian_dist(F1, F2, n)
    return D