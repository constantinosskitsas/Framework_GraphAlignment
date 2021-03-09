from random import random
import networkx as nx
import numpy as np
import argparse
import time
import os
import sys
import scipy.sparse as sp

try:
    import cPickle as pickle
except ImportError:
    import pickle
from scipy.sparse import csr_matrix


def check_with_identity(mb):
    count = 0
    for i in range(len(mb)):
        if i == mb[i]-1:
            count = count + 1
    return count/len(mb)


def accuracy(gma, gmb, mb, ma):
    nodes = len(gma)
    count = 0
    for i in range(nodes):
        if ma[i] == gma[i]:
            if (gmb[i]) == (mb[i]):
                count = count + 1
        else:
            print("mistake", ma[i], gma[i])
    print(count)
    return count / nodes


def accuracydiff(gma, gmb, mb, ma):
    nodes = len(ma) - 1
    nodes1 = len(gma)
    count = 0
    j = 0
    i = 0
    while i < nodes:
        if (ma[i] == gma[j]):
            if (gmb[j]) == (mb[i]):
                count = count + 1
                j = j+1
            i = i+1
        else:
            j = j+1
            # print("mistake", ma[i], gma[i])
    print(count)
    return count / nodes, count/nodes1


def accuracy2(gmb, mb):
    nodes = len(gmb)
    count = 0
    for i in range(nodes):
        print(gmb[i], mb[i])
        if (gmb[i]) == (mb[i]):
            count = count + 1
    print(count)
    return count / nodes


def split(Matching):
    Tempxx = (Matching[0])
    dd = len(Tempxx)

    split1 = np.zeros(len(Tempxx), int)
    split2 = np.zeros(len(Tempxx), int)
    for i in range(dd):
        tempMatching = Tempxx.pop()
        split1[i] = int(tempMatching[0])
        split2[i] = int(tempMatching[1])
    return split1, split2


def transformRAtoNormalALign(alignment_matrix):

    n_nodes = alignment_matrix.shape[0]
    sorted_indices = np.argsort(alignment_matrix)

    mb = np.zeros(1133)
    for node_index in range(n_nodes):
        target_alignment = node_index
        row, possible_alignments, possible_values = sp.find(
            alignment_matrix[node_index])
        node_sorted_indices = possible_alignments[possible_values.argsort()]
        mb[node_index] = node_sorted_indices[-1:]
    np.savetxt("results/matching.txt",
               np.vstack((range(1133), mb)).T, fmt="%i")
    mar = range(0, len(mb))
    return mar, mb
