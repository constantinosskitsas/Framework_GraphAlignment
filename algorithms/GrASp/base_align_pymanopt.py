import scipy.optimize as opt
import autograd.numpy as np
from . import grasp2 as fu
import sys
import matplotlib.pyplot as plt

import torch
import pymanopt
from pymanopt.manifolds import Stiefel
from pymanopt import Problem
from pymanopt.solvers import SteepestDescent

D1 = None
D2 = None
V1 = None
V2 = None
Cor1 = None
Cor2 = None
k_ = None


def main():
    graph_name = 'CA-GrQc'
    q = 100
    k = 20
    k_ = 20
    t = np.linspace(1, 50, q)

    V1 = np.load('eigens/' + graph_name + '/' +
                 graph_name + '_evectors_orig.npy')
    V2 = np.load('eigens/' + graph_name + '/5/evalues_2.npy')

    n = np.shape(V1)[0]

    D1 = np.load('eigens/' + graph_name + '/' +
                 graph_name + '_evalues_orig.npy')
    D2 = np.load('eigens/' + graph_name + '/5/evectors_2.npy')

    Cor1 = fu.calc_corresponding_functions(n, q, t, D1, V1)
    Cor2 = fu.calc_corresponding_functions(n, q, t, D2, V2)

    B = optimize_AB(Cor1, Cor2, n, V1, V2, D1, D2, k, k_)


def cost(X):
    mu = 0.132
    global D2
    global V1
    global V2
    global Cor1
    global Cor2
    global k_
    coup = (np.linalg.norm(Cor1.T@V1[:, 0:k_]-Cor2.T@V2[:, 0:k_]@X, 'fro'))**2
    res = (X.T @ np.diag(D2[0:k_]) @ X) ** 2
    diag_res = np.diagonal(res, offset=0, axis1=-1, axis2=-2)
    diag_res = np.sum(diag_res)
    sumres = np.sum(res)
    val = sumres - diag_res
    res = val+mu*coup
    return res


def optimize_AB(Cor11, Cor21, n, V11, V21, D11, D21, k):

    global D2
    global V1
    global V2
    global Cor1
    global Cor2
    global k_

    D2 = D21
    V1 = V11
    V2 = V21
    Cor1 = Cor11
    Cor2 = Cor21
    k_ = k

    manifold = Stiefel(k, k)
    x0 = init_x0(Cor1, Cor2, n, V1, V2, D1, D2, k)
    problem = Problem(manifold=manifold, cost=cost)

    # (3) Instantiate a Pymanopt solver

    solver = pymanopt.solvers.trust_regions.TrustRegions()  # maxtime=float('inf'))

    # let Pymanopt do the rest
    B = solver.solve(problem, x=x0)
    return B


def init_x0(Cor1, Cor2, n, V1, V2, D1, D2, k):

    B = np.identity(k)

    for i in range(0, k):
        thing1 = np.linalg.norm(Cor1.T@V1[:, i]-Cor2.T@V2[:, i])
        thing2 = np.linalg.norm(Cor1.T@V1[:, i]+Cor2.T@V2[:, i])
        if(thing1 > thing2):
            B[i, i] = -1
    return B


if __name__ == '__main__':
    main()
