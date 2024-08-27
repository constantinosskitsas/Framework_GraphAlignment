#Fugal Algorithm was provided by anonymous authors.
import numpy as np
from numpy import inf, nan
import scipy.sparse as sps
import scipy as sci
import numpy.matlib as matlib
from scipy.sparse.linalg import svds
from math import floor, log2
import math
import torch
import networkx as nx
import time
import matplotlib.pyplot as plt
from multiprocessing import Pool
from joblib import Parallel, delayed
from sklearn.metrics.pairwise import euclidean_distances
from algorithms.FUGAL.pred import convex_initTun,feature_extractionEV,feature_extraction,feature_extractionBM,eucledian_dist,convex_init

import numpy as np
import networkx as nx

from matplotlib import pyplot as plt
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score as ri
from sklearn.metrics import accuracy_score
import scipy
import pickle
import pandas as pd
import seaborn as sns
import ot
import warnings
import scipy.linalg as slg

from sklearn.cluster import SpectralClustering


def calculate_similarity_scores_from_matrices(G_A, G_B):
    # Step 1: Calculate degrees and normalize
    degrees_A = np.sum(G_A, axis=1)
    degrees_B = np.sum(G_B, axis=1)
    
    sum_degrees_A = np.sum(degrees_A)
    sum_degrees_B = np.sum(degrees_B)
    
    normalized_degrees_A = degrees_A / sum_degrees_A
    normalized_degrees_B = degrees_B / sum_degrees_B
    
    # Step 2: Compute similarity scores
    num_nodes_A = G_A.shape[0]
    num_nodes_B = G_B.shape[0]
    
    similarity_scores = np.zeros((num_nodes_A, num_nodes_B))
    
    for u in range(num_nodes_A):
        for v in range(num_nodes_B):
            d_A_u = normalized_degrees_A[u]
            d_B_v = normalized_degrees_B[v]
            similarity_scores[u, v] = min(d_A_u, d_B_v) / max(d_A_u, d_B_v)
    
    return similarity_scores

def create_L(A, B, lalpha=10000, mind=None, weighted=True):
    n = A.shape[0]
    m = B.shape[0]

    if lalpha is None:
        return sps.csr_matrix(np.ones((n, m)))

    a = A.sum(1)
    b = B.sum(1)
    # print(a)
    # print(b)
    DegA=A.sum()
    DegB=B.sum()
    # a_p = [(i, m[0,0]) for i, m in enumerate(a)]
    a_p = list(enumerate(a))
    a_p.sort(key=lambda x: x[1])

    # b_p = [(i, m[0,0]) for i, m in enumerate(b)]
    b_p = list(enumerate(b))
    b_p.sort(key=lambda x: x[1])

    ab_m = [0] * n
    s = 0
    e = floor(lalpha * log2(m))
    a=a/DegA
    b=b/DegB
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
            d= min(a[i],b[j])/max(a[i],b[j])
            #d = 1 - abs(a[i]-b[j]) / a[i]+b[j]
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

    return sps.csr_matrix((lw, (li, lj)), shape=(n, m))

def main(data, iter,simple,mu):
    print("Fugal2")
    torch.set_num_threads(40)
    print("Fugal2")
    #torch.set_num_threads(40)
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
        # Step 1: Calculate degrees and normalize

    
    #Src = degrees_A / sum_degrees_A
    #Tar = degrees_B / sum_degrees_B
    D=calculate_similarity_scores_from_matrices(Src,Tar)
    D1=create_L(Src,Tar)
    #print(D)
    #print(D1)
    #dense_array = D.toarray()
    n1 = Tar.shape[0]
    n2 = Src.shape[0]
    n = max(n1, n2)
    Src1=nx.from_numpy_array(Src)
    Tar1=nx.from_numpy_array(Tar)

    Src2=nx.normalized_laplacian_matrix(Src1).todense()
    Tar2=nx.normalized_laplacian_matrix(Tar1).todense()
    
    A = torch.tensor((Src2), dtype = torch.float64)
    B = torch.tensor((Tar2), dtype = torch.float64)
    #print(A)
    #F1= feature_extractionEV(Src1)
    #F2= feature_extractionEV(Tar1)
    F1= feature_extraction(Src1,simple)
    F2= feature_extraction(Tar1,simple)
    K = eucledian_dist(F1, F2, n)
    D=torch.tensor(D, dtype = torch.float64)
    K = torch.tensor(K, dtype = torch.float64)
    P1=convex_initTun(A, B, D,K, mu, iter)
    return P1
import numpy as np
from scipy.optimize.linesearch import scalar_search_armijo
from ot.lp import emd
from scipy.sparse.csgraph import shortest_path
def solve_1d_linesearch_quad_funct(a,b,c):
    # solve min f(x)=a*x**2+b*x+c sur 0,1
    f0=c
    df0=b
    f1=a+f0+df0
    if a>0: # convex
        minimum=min(1,max(0,-b/(2*a)))
        return minimum
    else:
        if f0>f1:
            return 1
        else:
            return 0
def line_search_armijo(f, xk, pk, gfk, old_fval, args=(), c1=1e-4, alpha0=0.99):
    xk = np.atleast_1d(xk)
    fc = [0]
    def phi(alpha1):
        fc[0] += 1
        return f(xk + alpha1 * pk, *args)
    if old_fval is None:
        phi0 = phi(0.)
    else:
        phi0 = old_fval
    derphi0 = np.sum(pk * gfk)  # Quickfix for matrices
    alpha, phi1 = scalar_search_armijo(
        phi, phi0, derphi0, c1=c1, alpha0=alpha0)
    return alpha, fc[0], phi1
def do_linesearch(cost,G,deltaG,Mi,f_val,amijo=True,C1=None,C2=None,reg=None,Gc=None,constC=None,M=None):
    if amijo:
        alpha, fc, f_val = line_search_armijo(cost, G, deltaG, Mi, f_val)
    else:
        dot1=np.dot(C1,deltaG)
        dot12=dot1.dot(C2) # C1 dt C2
        a=-2*reg*np.sum(dot12*deltaG) #-2*alpha*<C1 dt C2,dt> si qqlun est pas bon c'est lui
        b=np.sum((M+reg*constC)*deltaG)-2*reg*(np.sum(dot12*G)+np.sum(np.dot(C1,G).dot(C2)*deltaG))
        c=cost(G) #f(xt)
        alpha=solve_1d_linesearch_quad_funct(a,b,c)
        fc=None
        f_val=cost(G+alpha*deltaG)
    return alpha,fc,f_val
def cg(a, b, M, reg, f, df, G0=None, numItermax=500, stopThr=1e-09, verbose=False,log=False,amijo=True,C1=None,C2=None,constC=None):
    loop = 1
    if log:
        log = {'loss': [],'delta_fval': []}
    if G0 is None:
        G = np.outer(a, b)
    else:
        G = G0
    def cost(G):
        return np.sum(M * G) + reg * f(G)
    f_val = cost(G) #f(xt)
    if log:
        log['loss'].append(f_val)
    it = 0
    if verbose:
        print('{:5s}|{:12s}|{:8s}'.format(
            'It.', 'Loss', 'Delta loss') + '\n' + '-' * 32)
        print('{:5d}|{:8e}|{:8e}'.format(it, f_val, 0))
    G_prev = G
    while loop:
        it += 1
        old_fval = f_val
        #G=xt
        # problem linearization
        Mi = M + reg * df(G) #Gradient(xt)
        # set M positive
        Mi += Mi.min()
        # solve linear program
        Gc = emd(a, b, Mi) #st
        deltaG = Gc - G #dt
        # argmin_alpha f(xt+alpha dt)
        alpha, fc, f_val = do_linesearch(cost=cost,G=G,deltaG=deltaG,Mi=Mi,f_val=f_val,amijo=amijo,constC=constC,C1=C1,C2=C2,reg=reg,Gc=Gc,M=M)
        if alpha is None or np.isnan(alpha) :
            break
            # raise NonConvergenceError('Alpha n a pas été trouvé')
        else:
            G = G + alpha * deltaG #xt+1=xt +alpha dt
        # test convergence
        if it >= numItermax:
            loop = 0
        delta_fval = (f_val - old_fval)
        #delta_fval = (f_val - old_fval)/ abs(f_val)
        #print(delta_fval)
        if abs(delta_fval) < stopThr:
            loop = 0
        if log:
            log['loss'].append(f_val)
            log['delta_fval'].append(delta_fval)
        if verbose:
            if it % 20 == 0:
                print('{:5s}|{:12s}|{:8s}'.format(
                    'It.', 'Loss', 'Delta loss') + '\n' + '-' * 32)
            print('{:5d}|{:8e}|{:8e}|{:5e}'.format(it, f_val, delta_fval,alpha))
    if log:
        return G, log
    else:
        return G
def tensor_product(constC,hC1,hC2,T):
    A=-np.dot(hC1, T).dot(hC2.T)
    tens = constC+A
    return tens
def gwloss(constC,hC1,hC2,T):
    tens=tensor_product(constC,hC1,hC2,T)
    return np.sum(tens*T)
def gwggrad(constC,hC1,hC2,T):
    return 2*tensor_product(constC,hC1,hC2,T)
def init_matrix(C1,C2,p,q,loss_fun='square_loss'):
    if loss_fun == 'square_loss':
        def f1(a):
            return a**2
        def f2(b):
            return b**2
        def h1(a):
            return a
        def h2(b):
            return 2*b
    constC1 = np.dot(np.dot(f1(C1), p.reshape(-1, 1)),
                     np.ones(len(q)).reshape(1, -1))
    constC2 = np.dot(np.ones(len(p)).reshape(-1, 1),
                     np.dot(q.reshape(1, -1), f2(C2).T))
    constC=constC1+constC2
    hC1 = h1(C1)
    hC2 = h2(C2)
    return constC,hC1,hC2
def gw_lp(C1,C2,p,q,loss_fun='square_loss',alpha=1,amijo=True,**kwargs):
    constC,hC1,hC2=init_matrix(C1,C2,p,q,loss_fun)
    M=np.zeros((C1.shape[0],C2.shape[0]))
    G0=p[:,None]*q[None,:]
    def f(G):
        return gwloss(constC,hC1,hC2,G)
    def df(G):
        return gwggrad(constC,hC1,hC2,G)
    return cg(p,q,M,alpha,f,df,G0,amijo=amijo,constC=constC,C1=C1,C2=C2,**kwargs)
def fgw_lp(M,C1,C2,p,q,loss_fun='square_loss',alpha=1,amijo=True,G0=None,**kwargs):
    constC,hC1,hC2=init_matrix(C1,C2,p,q,loss_fun)
    if G0 is None:
        G0=p[:,None]*q[None,:]
    def f(G):
        return gwloss(constC,hC1,hC2,G)
    def df(G):
        return gwggrad(constC,hC1,hC2,G)
    return cg(p,q,M,alpha,f,df,G0,amijo=amijo,C1=C1,C2=C2,constC=constC,**kwargs)
def fgw(D,A1,A2):
    n=A1.shape[0]
    C1=shortest_path(A1)
    C2=shortest_path(A2)
    ones = np.ones(n, dtype = np.float64)
    return fgw_lp(D,C1,C2,ones,ones)
def gw(A1,A2):
    n=A1.shape[0]
    C1=shortest_path(A1)
    C2=shortest_path(A2)
    ones = np.ones(n, dtype = np.float64)
    return gw_lp(C1,C2,ones,ones)