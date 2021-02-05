import scipy
import numpy as np


def bipartite_matching(A, nzi, nzj, nzv):
    # return bipartite_matching_primal_dual(bipartite_matching_setup(A,nzi,nzj,nzv))
    rp, ci, ai, [], m, n = bipartite_matching_setup(A, nzi, nzj, nzv)
    return bipartite_matching_primal_dual(rp, ci, ai, [], m, n)


def bipartite_matching_primal_dual(rp, ci, ai, tripi, m, n):
    # variables used for the primal-dual algorithm
    # normalize ai values # updated on 2-19-2019
    ai = ai/np.amax(abs(ai))
    alpha = np.zeros(m)
    bt = np.zeros(m+n)  # beta
    queue = np.zeros(m, int)
    t = np.zeros(m+n, int)
    match1 = np.zeros(m, int)
    match2 = np.zeros(m+n, int)
    tmod = np.zeros(m+n, int)
    ntmod = 0

    # initialize the primal and dual variables
    for i in range(1, m):
        for rpi in range(rp[i], rp[i+1]):
            if ai[rpi] > alpha[i]:
                alpha[i] = ai[rpi]

    # dual variables (bt) are initialized to 0 already
    # match1 and match2 are both 0, which indicates no matches

    i = 1
    while i < m:
        for j in range(1, ntmod+1):
            t[tmod[j]] = 0
        ntmod = 0
        # add i to the stack
        head = 1
        tail = 1
        queue[head] = i
        while head <= tail and match1[i] == 0:
            k = queue[head]
            for rpi in range(rp[k], rp[k+1]):
                j = ci[rpi]
                if ai[rpi] < alpha[k] + bt[j] - 1e-8:
                    continue

                if t[j] == 0:
                    tail = tail+1
                    if tail <= m:
                        queue[tail] = match2[j]
                    t[j] = k
                    ntmod = ntmod+1
                    tmod[ntmod] = j
                    if match2[j] < 1:
                        while j > 0:
                            match2[j] = t[j]
                            k = t[j]
                            temp = match1[k]
                            match1[k] = j
                            j = temp
                        break
            head = head+1
        if match1[i] < 1:
            theta = np.inf
            for j in range(1, head):
                t1 = queue[j]
                for rpi in range(rp[t1], rp[t1+1]):
                    t2 = ci[rpi]
                    if t[t2] == 0 and alpha[t1] + bt[t2] - ai[rpi] < theta:
                        theta = alpha[t1] + bt[t2] - ai[rpi]
            for j in range(1, head):
                alpha[queue[j]] -= theta
            for j in range(1, ntmod+1):
                bt[tmod[j]] += theta
            continue
        i = i+1
    val = 0
    for i in range(1, m):
        for rpi in range(rp[i], rp[i+1]):
            if ci[rpi] == match1[i]:
                val = val+ai[rpi]
    noute = 0
    for i in range(1, m):
        if match1[i] <= n:
            noute = noute+1
    return m, n, val, noute, match1


def bipartite_matching_setup(A, nzi, nzj, nzv):
    #(nzi,nzj,nzv) = bipartite_matching_setup_phase1(A,nzi,nzj,nzv)
    (nzi, nzj, nzv) = bipartite_matching_setup_phase1(A)
    nedges = len(nzi)
    (m, n) = np.shape(A)
    m = m+1
    n = n+1
    rp = np.ones(m+1, int)  # csr matrix with extra edges
    ci = np.zeros(nedges+m, int)
    ai = np.zeros(nedges+m)

    rp[0] = 0
    rp[1] = 0
    for i in range(1, nedges):
        rp[nzi[i]+1] = rp[nzi[i]+1]+1
    rp = np.cumsum(rp)

    for i in range(1, nedges):
        ai[rp[nzi[i]]+1] = nzv[i]
        ci[rp[nzi[i]]+1] = nzj[i]
        rp[nzi[i]] = rp[nzi[i]]+1

    for i in range(1, m):  # add the extra edges
        ai[rp[i]+1] = 0
        ci[rp[i]+1] = n+i
        rp[i] = rp[i]+1

    # restore the row pointer array
    for i in range(m-1, 0, -1):
        rp[i+1] = rp[i]
    rp[1] = 0
    rp = rp+1

    # check for duplicates in the data
    colind = np.zeros(m+n, int)
    for i in range(1, m):
        for rpi in range(rp[i], rp[i+1]):
            if colind[ci[rpi]] == 1:
                print("bipartite_matching:duplicateEdge")
        colind[ci[rpi]] = 1

        for rpi in range(rp[i], rp[i+1]):
            colind[ci[rpi]] = 0
    return rp, ci, ai, [], m, n


def bipartite_matching_setup_phase1(A, nzi, nzj, nzv):
    temp = len(nzi)+1
    nzi1 = np.zeros(temp, int)
    nzj1 = np.zeros(temp, int)
    nzv1 = np.zeros(temp, float)
    for i in range(1, temp):
        nzi1[i] = nzi[i-1]
        nzj1[i] = nzj[i - 1]
        nzv1[i] = nzv[i - 1]
    nzi1 = nzi1+1
    nzj1 = nzj1+1
    return (nzi1, nzj1, nzv1)


def bipartite_matching_setup_phase1(A):
    nzi, nzj = scipy.sparse.csr_matrix.nonzero(A)
    temp = len(nzi) + 1
    nzi1 = np.zeros(temp, int)
    nzj1 = np.zeros(temp, int)
    nzv1 = np.zeros(temp, float)
    for i in range(1, temp):
        nzi1[i] = nzi[i - 1]
        nzj1[i] = nzj[i - 1]
        nzv1[i] = A[nzi1[i], nzj1[i]]
    nzi1 = nzi1 + 1
    nzj1 = nzj1 + 1

    return (nzi1, nzj1, nzv1)


def edge_list(m, n, weight, cardinality, match):
    m1 = np.zeros(cardinality, int)
    m2 = np.zeros(cardinality, int)
    noute = 0
    for i in range(1, m):
        if match[i] <= n:
            m1[noute] = i
            m2[noute] = match[i]
            noute = noute+1
    return m1, m2


def bipartite_matching_setupc(x, ei, ej, m, n):
    (nzi, nzj, nzv) = (ei, ej, x)
    nedges = len(nzi)

    rp = np.ones(m+1, int)  # csr matrix with extra edges
    ci = np.zeros(nedges+m, int)
    ai = np.zeros(nedges+m)
    tripi = np.zeros(nedges+m, int)
    # 1. build csr representation with a set of extra edges from vertex i to
    # vertex m+i
    rp[0] = 0
    rp[1] = 0
    for i in range(1, nedges):
        rp[nzi[i]+1] = rp[nzi[i]+1]+1

    rp = np.cumsum(rp)
    for i in range(1, nedges):
        tripi[rp[nzi[i]]+1] = i
        ai[rp[nzi[i]]+1] = nzv[i]
        ci[rp[nzi[i]]+1] = nzj[i]
        rp[nzi[i]] = rp[nzi[i]]+1
    for i in range(1, m):  # add the extra edges
        tripi[rp[i]+1] = -1
        ai[rp[i]+1] = 0
        ci[rp[i]+1] = n+i
        rp[i] = rp[i]+1

    # restore the row pointer array
    for i in range(m, 0, -1):
        rp[i+1] = rp[i]
    rp[1] = 0
    rp = rp+1
    rp[0] = 0

    # check for duplicates in the data
    colind = np.zeros(m+n)

    for i in range(1, m):
        for rpi in range(rp[i], rp[i+1]-1):
            if colind[ci[rpi]] == 1:
                print("bipartite_matching:duplicateEdge")
        colind[ci[rpi]] = 1

        for rpi in range(rp[i], rp[i+1]-1):
            colind[ci[rpi]] = 0

    return rp, ci, ai, tripi, m, n
