#Fugal Algorithm was provided by anonymous authors.
import numpy as np
from numpy import inf, nan
import scipy.sparse as sps
import scipy as sci
from math import floor, log2
import math
import torch
from tqdm.auto import tqdm
import networkx as nx
import time
import matplotlib.pyplot as plt
from multiprocessing import Pool
from sklearn.metrics.pairwise import euclidean_distances
from algorithms.FUGAL.pred import feature_extraction,eucledian_dist,convex_initQAP
#from pred import feature_extraction,eucledian_dist,convex_init
def create_L(A, B, lalpha=1, mind=None, weighted=True):
    n = A.shape[0]
    m = B.shape[0]

    if lalpha is None:
        return sps.csr_matrix(np.ones((n, m)))

    a = A.sum(1)
    b = B.sum(1)
    # print(a)
    # print(b)

    # a_p = [(i, m[0,0]) for i, m in enumerate(a)]
    a_p = list(enumerate(a))
    a_p.sort(key=lambda x: x[1])

    # b_p = [(i, m[0,0]) for i, m in enumerate(b)]
    b_p = list(enumerate(b))
    b_p.sort(key=lambda x: x[1])

    ab_m = [0] * n
    s = 0
    e = floor(lalpha * log2(m))
    for ap in a_p:
        while(e < m and
              abs(b_p[e][1] - ap[1]) <= abs(b_p[s][1] - ap[1])
              ):
            e += 1
            s += 1
        ab_m[ap[0]] = [bp[0] for bp in b_p[s:e]]

    # print(ab_m)

    li = []
    lj = []
    lw = []
    for i, bj in enumerate(ab_m):
        for j in bj:
            # d = 1 - abs(a[i]-b[j]) / a[i]
            #d = 1 - abs(a[i]-b[j]) / max(a[i], b[j])
            d=abs(a[i]-b[j]) / max(a[i], b[j])
            #d=abs(a[i]-b[j])
            if mind is None:
                if d > 0:
                    li.append(i)
                    lj.append(j)
                    lw.append(d)
            else:
                li.append(i)
                lj.append(j)
                lw.append(mind if d <= 0 else d)
                # lw.append(0.0 if d <= 0 else d)
                # lw.append(d)

                # print(len(li))
                # print(len(lj))
                # print(len(lj))

    return sps.csr_matrix((lw, (li, lj)), shape=(n, m))

def decomposeN_laplacian(A):

    #  adjacency matrix

    Deg = np.diag((np.sum(A, axis=1)))

    n = np.shape(Deg)[0]

    Deg = sci.linalg.fractional_matrix_power(Deg, -0.5)

    L = np.identity(n) - Deg @ A @ Deg
    #P=np.linalg.inv(Deg)@ A@np.linalg.inv(Deg)
    #L=np.identity(n) - P
    #L= A+Deg
   # print((sci.fractional_matrix_power(Deg, -0.5) * A * sci.fractional_matrix_power(Deg, -0.5)))
    # '[V1, D1] = eig(L1);

    D, V = np.linalg.eigh(L)

    return [D, V]

def main(data, iter,simple,mu):
    print("QAP")
    dtype = np.float64
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
    Src1=nx.from_numpy_array(Src)
    Tar1=nx.from_numpy_array(Tar)
    A = torch.tensor((Src), dtype = torch.float64)
    B = torch.tensor((Tar), dtype = torch.float64)
    P1=convex_initQAP(A, B, iter)
    return P1
