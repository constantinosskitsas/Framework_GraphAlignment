import numpy as np
import math
# import torch
# import torch_geometric as tg
# import torch_geometric.data
# import torch_geometric.datasets
from tqdm.auto import tqdm
# from torch_geometric.data import Data

import networkx as nx

import time
import heapq
import itertools as it
import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')
import os
import random
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import sys
import matplotlib.pyplot as plt

def EC(A, B, ma, mb):
    adj1 = A[ma][:, ma]
    adj2 = B[mb][:, mb]
    comb = adj1 + adj2

    intersection = np.sum(comb == 2)

    return intersection / np.sum(A == 1)


def ICS(A, B, ma, mb):
    adj1 = A[ma][:, ma]
    adj2 = B[mb][:, mb]
    comb = adj1 + adj2

    intersection = np.sum(comb == 2)
    induced = np.sum(adj2 == 1)

    return intersection / induced


def S3(A, B, ma, mb):
    adj1 = A[ma][:, ma]
    adj2 = B[mb][:, mb]
    comb = adj1 + adj2

    intersection = np.sum(comb == 2)
    induced = np.sum(adj2 == 1)
    denom = np.sum(A == 1) + induced - intersection

    return intersection / denom


def jacc(A, B, ma, mb):
    adj1 = A[ma][:, ma]
    adj2 = B[mb][:, mb]
    comb = adj1 + adj2

    intersection = np.sum(comb == 2)
    union = np.sum(A == 1) + np.sum(B == 1) - intersection

    return intersection / union

def eval_align(ma, mb, gmb):

    try:
        gmab = np.arange(gmb.size)
        gmab[ma] = mb
        gacc = np.mean(gmb == gmab)

        mab = gmb[ma]
        acc = np.mean(mb == mab)

    except Exception:
        mab = np.zeros(mb.size, int) - 1
        gacc = acc = -1.0
    alignment = np.array([ma, mb, mab]).T
    alignment = alignment[alignment[:, 0].argsort()]
    return gacc, acc, alignment


def ged(A, B, ma, mb):
    n = ma.size
    P = np.zeros((n, n))
    for i in range(n):
        P[i][mb[i]] = 1
    X = np.matmul(A, P) - np.matmul(P, B)
    norm = np.linalg.norm(X, 'fro')
    return norm*norm/2

def ged_rmse(v1, v2):
    n = len(v1)
    res = 0
    for i in range(n):
        res += (v1[i] - v2[i])**2
    res /= n
    return math.sqrt(res)

def avg(lb, ub):
    res = []
    for i in range(len(lb)):
        res.append((lb[i] + ub[i])/2)
    return res

def rmse(lb, ub, pred):
    truth = avg(lb, ub)
    rmse_error = ged_rmse(pred, truth)
    return rmse_error
