

import numpy as np
def edgelist_to_adjmatrix1(edgeList_file):
    true_alignments = np.loadtxt(edgeList_file)
    n = int(np.amax(true_alignments)) + 1
    e = np.shape(true_alignments)[0]
    a = np.zeros((n, n), dtype=int)
    #
    # make adjacency matrix A1
    for i in range(e):
        n1 = int(true_alignments[i, 0])  # +1
        n2 = int(true_alignments[i, 1])  # +1
        a[n1][n2] = 1.0
        a[n2][n1] = 1.0
    return a

def gt(edgeList_file, gma=None):
    true_alignments = np.loadtxt(edgeList_file)
    gman = int(np.amax(true_alignments)) + 1
    gmbn = int(np.amax(true_alignments)) + 1
    gma = np.zeros(gman)
    gmb = np.zeros(gmbn)
    e = np.shape(true_alignments)[0]
    #
    # make adjacency matrix A1
    for i in range(gman):
        gma[i] = (int)(true_alignments[i][0])+1
        gmb[i] = (int)(true_alignments[i][1])+1
    return gma, gmb
def gt1(edgeList_file, gma=None):
    true_alignments = np.loadtxt(edgeList_file)
    gman = int(np.amax(true_alignments)) + 1
    gmbn = int(np.amax(true_alignments)) + 1
    gma = np.zeros(gman)
    gmb = np.zeros(gmbn)
    e = np.shape(true_alignments)[0]
    #
    # make adjacency matrix A1
    for i in range(gman):
        a=(int)(true_alignments[i][0])
        b=(int)(true_alignments[i][1])
        gma[i]=i
        gmb[a]=b
    return gma, gmb



def edgelist_to_adjmatrixR(edgeList_file, edgeList_file1):
    true_alignments = np.loadtxt(edgeList_file)
    true_alignments1 = np.loadtxt(edgeList_file1)
    n = int(np.amax(true_alignments)) + 1
    m = int(np.amax(true_alignments1)) + 1
    e = np.shape(true_alignments)[0]
    a = np.zeros((n + m, n + m), dtype=int)
    # make adjacency matrix A1
    print("n ", n)
    for i in range(e):
        n1 = int(true_alignments[i, 0])  # +1
        n2 = int(true_alignments[i, 1])  # +1
        a[n1, n2] = 1.0
        a[n2, n1] = 1.0
    e = np.shape(true_alignments1)[0]
    for i in range(e):
        n1 = int(true_alignments1[i, 0])  # +1
        n2 = int(true_alignments1[i, 1])  # +1
        a[n1 + n, n2 + n] = 1.0
        a[n2 + n, n1 + n] = 1.0
    #a = remove_edges_directed(a,n)
    return a