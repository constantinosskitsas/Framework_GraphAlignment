import numpy as np

# /**
# * n the number of nodes
# * m the number of nodes
# * nedges the number of edges
# * v1 is the source for each of the nedges
# * v2 is the target for each of the nedges
# * weight is the weight of each of the nedges
# * mi is a vector saying which of v1 and v2 are used, length >= nedges
# */


def intmatch(n,  m, nedges,  v1,  v2,  weight):

    l1 = np.zeros(n)
    l2 = np.zeros(n+m)
    s = np.zeros(n+m, int)
    t = np.zeros(n+m, int)
    offset = np.zeros(n, int)
    deg = np.ones(n, int)
    listt = np.zeros(nedges + n, int)
    index = np.zeros(nedges+n, int)
    w = np.zeros(nedges + n)
    match1 = np.zeros(n, int)
    match2 = np.zeros(n+m, int)

    ntmod = 0
    tmod = np.zeros(n+m, int)

    for i in range(nedges):
        deg[v1[i]] += 1
    for i in range(1, n):
        offset[i] = offset[i-1] + deg[i-1]
    deg = np.zeros(n, int)
    for i in range(nedges):
        listt[offset[v1[i]] + deg[v1[i]]] = v2[i]
        w[offset[v1[i]] + deg[v1[i]]] = weight[i]
        index[offset[v1[i]] + deg[v1[i]]] = i
        deg[v1[i]] += 1

    for i in range(n):
        listt[offset[i] + deg[i]] = m + i
        w[offset[i] + deg[i]] = 0
        index[offset[i] + deg[i]] = -1
        deg[i] += 1

    for i in range(n):
        l1[i] = 0
        for j in range(deg[i]):
            if (w[offset[i]+j] > l1[i]):
                l1[i] = w[offset[i] + j]

    match1 = np.zeros(n, int)-1
    l2 = np.zeros(n+m)
    match2 = np.zeros(n+m, int)-1
    t = np.zeros(n+m, int)-1

    for i in range(n):
        for j in range(ntmod):
            t[tmod[j]] = -1
        ntmod = 0
        p = q = 0
        s[0] = i
        while(p <= q):
            if (match1[i] >= 0):
                break
            k = s[p]
            for r in range(deg[k]):
                j = listt[offset[k] + r]
                if (w[offset[k] + r] < l1[k] + l2[j] - 1e-8):
                    continue
                if (t[j] < 0):
                    q += 1
                    s[q] = match2[j]
                    t[j] = k
                    tmod[ntmod] = j
                    ntmod += 1
                    if (match2[j] < 0):
                        while(j >= 0):
                            k = match2[j] = t[j]
                            p = match1[k]
                            match1[k] = j
                            j = p
                        break
            p += 1

        if (match1[i] < 0):
            al = 1e20
            for j in range(p):
                t1 = s[j]
                for k in range(deg[t1]):
                    t2 = listt[offset[t1] + k]
                    if (t[t2] < 0 and l1[t1] + l2[t2] - w[offset[t1] + k] < al):
                        al = l1[t1] + l2[t2] - w[offset[t1] + k]
            for j in range(p):
                l1[s[j]] -= al
            # // for (j=0
            #         j < n + m
            #         j++) if (t[j] >= 0) l2[j] += al
            for j in range(ntmod):
                l2[tmod[j]] += al

            i -= 1
            continue

    ret = 0
    for i in range(n):
        for j in range(deg[i]):
            if listt[offset[i] + j] == match1[i]:
                ret += w[offset[i] + j]

    mi = np.zeros(nedges)
    for i in range(n):
        if (match1[i] < m):
            for j in range(deg[i]):
                if (listt[offset[i] + j] == match1[i]):
                    mi[index[offset[i]+j]] = 1
    return ret, mi


# /**
# * M: rows of Q
# * N: columns of Q
# * Qp: column pointers for Q(Q is compressed column)
# length >= N+1
# * Qr: row indices for Q
# length >= Qp[N]
# * Qv: values for Q
# length >= Qp[N]
# * m: rows of bipartite L
# * n: columns of bipartite L
# * nedges: number of edges in L(nedges < M)
# * li: first index of edges in L
# length >= nedges
# * lj: second index of edges in L
# length >= nedges
# * q[out]: output variable for the biggest maximum match sum of a column of Q
# *      length >= N
# * mi[out]: the first index of selected edges in Q
# length >= Qp[N]
# * mj[out]: the second index of selected edges in Q
# length >= Qp[N]
# * outmedges[out]: the number of edges in mi and mj used
# */
def column_maxmatchsum(M, N, Qp, Qr, Qv, m, n, nedges, li, lj):
    # print(M, N, m, n, nedges)
    # print(Qp)
    # print(Qp.shape)
    # print(Qr)
    # print(Qr.shape)
    # print(Qv)
    # print(Qv.shape)
    # print(li)
    # print(li.shape)
    # print(lj)
    # print(lj.shape)

    q = np.zeros(N)
    mi = []
    mj = []

    ili = li - 1
    ilj = lj - 1

    lwork1 = np.zeros(m, int)
    lwork2 = np.zeros(n, int)

    max_col_nonzeros = 0
    for j in range(N):
        col_nonzeros = Qp[j+1]-Qp[j]
        if (col_nonzeros >= max_col_nonzeros):
            max_col_nonzeros = col_nonzeros

    se1 = np.zeros(max_col_nonzeros, int)
    se2 = np.zeros(max_col_nonzeros, int)
    sw = np.zeros(max_col_nonzeros)
    sqi = np.zeros(max_col_nonzeros, int)
    sqj = np.zeros(max_col_nonzeros, int)
    smi = np.zeros(max_col_nonzeros, int)

    lind1 = np.zeros(m) - 1
    lind2 = np.zeros(n) - 1

    for j in range(N):
        smalledges = 0
        nsmall1 = 0
        nsmall2 = 0

        for nzi in range(Qp[j], Qp[j+1]):

            i = Qr[nzi]
            v1 = ili[i]
            v2 = ilj[i]
            sv1 = sv2 = -1
            if (lind1[v1] < 0):
                sv1 = nsmall1
                lind1[v1] = sv1
                lwork1[sv1] = v1
                nsmall1 += 1
            else:
                sv1 = lind1[v1]
            if (lind2[v2] < 0):
                sv2 = nsmall2
                lind2[v2] = sv2
                lwork2[sv2] = v2
                nsmall2 += 1
            else:
                sv2 = lind2[v2]

            se1[smalledges] = sv1
            se2[smalledges] = sv2
            sw[smalledges] = Qv[nzi]
            sqi[smalledges] = i
            sqj[smalledges] = j
            smalledges += 1

        if (smalledges == 0):
            q[j] = 0.0
            continue

        q[j], smi = intmatch(nsmall1, nsmall2, smalledges, se1, se2, sw)

        for k in range(smalledges):
            if (smi[k] > 0):
                mi.append(sqi[k])
                mj.append(sqj[k])

        for k in range(nsmall1):
            lind1[lwork1[k]] = -1

        for k in range(nsmall2):
            lind2[lwork2[k]] = -1

    return q, np.array(mi), np.array(mj)


if __name__ == "__main__":
    # M = 6
    # N = 5
    M = 12
    N = 12
    nedges = 31
    li = np.array([1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 6])
    lj = np.array([1, 2, 3, 4, 1, 2, 1, 3, 1, 4, 5, 2])
    m = 6
    n = 5
    # Qp = np.array([0, 6, 8, 10, 11, 12, 5, 7, 9, 5, 7, 9, 5, 7,
    #             9, 2, 3, 4, 12, 1, 2, 3, 4, 1, 2, 3, 4, 1, 1, 1, 5])
    # Qr = np.array([0, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4,
    #              5, 5, 5, 5, 6, 7, 7, 7, 8, 9, 9, 9, 10, 11, 12, 12])
    Qv = np.array(
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
         0.5, 0.5,
         0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    )
    Qp = np.array([0, 5, 8, 11, 14, 18, 19, 22, 23, 26, 27, 28, 30])
    Qr = np.array([5, 7, 9, 10, 11, 4, 6, 8, 4, 6, 8, 4, 6, 8,
                   1, 2, 3, 11, 0, 1, 2, 3, 0, 1, 2, 3, 0, 0, 0, 4])
    print(column_maxmatchsum(M, N, Qp, Qr, Qv, m, n, nedges, li, lj))

# [2., 0.5, 0.5, 0.5, 1., 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
# [6,  8, 10, 11,  5,  5,  5,  3, 12,  1,  2,  1,  2,  1,  1,  1]
# [1,  1,  1,  1,  2,  3,  4,  5,  5,  6,  7,  8,  9, 10, 11, 12]
