import numpy as np

# def print(*args):
#     pass


def intmatch(n, m, nedges, v1, v2, weight):
    print(n, m, nedges)
    print(v1)
    print(v2)
    print(weight)

    n = n + 1
    m = m + 1
    nedges = nedges + 1
    l1 = np.zeros(n, float)
    l2 = np.zeros(n + m - 1, float)
    s = np.ones(n + m - 1, int)
    t = -1 * np.ones(n + m - 1, int)
    offset = np.zeros(n, int)
    deg = np.ones(n, int)
    list = np.zeros(nedges + n - 1, int)
    index = np.zeros(nedges + n - 1, int)
    w = np.zeros(nedges + n - 1, float)
    match1 = -1 * np.ones(n, int)
    match2 = -1 * np.ones(n + m - 1, int)
    tmod = np.zeros(m + n - 1, int)
    ntmod = 0

    for i in range(1, nedges):
        deg[v1[i]] += 1

    for i in range(2, n):
        offset[i] = offset[i - 1] + deg[i - 1]
    offset = offset + 1
    deg = np.zeros(n, int)
    for i in range(1, nedges):
        list[offset[v1[i]] + deg[v1[i]]] = v2[i]
        w[offset[v1[i]] + deg[v1[i]]] = weight[i]
        index[offset[v1[i]] + deg[v1[i]]] = i
        deg[v1[i]] += 1

    for i in range(1, n):
        list[offset[i] + deg[i]] = m + i - 1
        w[offset[i] + deg[i]] = 0
        index[offset[i] + deg[i]] = -1
        deg[i] += 1
    for i in range(1, n):
        for j in range(0, deg[i] - 1):
            if w[offset[i] + j] > l1[i]:
                l1[i] = w[offset[i] + j]
    i = 1
    while i <= n - 1:
        for j in range(1, ntmod+1):
            t[tmod[j]] = -1
        ntmod = 0
        p = 1
        q = 1
        s[1] = i
        while p <= q:
            if match1[i] >= 1:
                break
            k = s[p]
            for r in range(0, deg[k]):
                j = list[offset[k] + r]
                val1totest = w[offset[k] + r]
                val2totest = l1[k] + l2[j] - 1e-8
                if w[offset[k] + r] < val2totest:
                    continue
                if t[j] < 0:
                    q = q + 1
                    s[q] = match2[j]
                    t[j] = k
                    tmod[ntmod] = j
                    if match2[j] < 0:
                        while j >= 1:
                            match2[j] = t[j]
                            k = match2[j]
                            # k = match2[j] = t[j]
                            p = match1[k]
                            match1[k] = j
                            j = p
                        break
            p += 1
        p -= 1
        if match1[i] < 0:
            al = 1e20
            for j in range(1, p):
                t1 = s[j]
                for k in range(0, deg[t1]):
                    t2 = list[offset[t1] + k]
                    if t[t2] < 0 and ((l1[t1] + l2[t2] - w[offset[t1] + k]) < al):
                        al = l1[t1] + l2[t2] - w[offset[t1] + k]
            for j in range(1, p):
                vtemp = s[j]
                vvtemp = l1[s[j]]
                l1[s[j]] -= al
                vvtemp = l1[s[j]]
            for j in range(1, ntmod):
                l2[tmod[j]] += al
        else:
            i += 1
    ret = 0.0
    for i in range(1, n):
        for j in range(0, deg[i] - 1):
            if list[offset[i] + j] == match1[i]:
                ret += w[offset[i] + j]

    mi = np.zeros(nedges, int)
    for i in range(1, n):
        if match1[i] <= m:
            for j in range(0, deg[i] - 1):
                if list[offset[i] + j] == match1[i]:
                    mi[index[offset[i] + j]] = 1

    return ret, mi


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

    # with open('Qp.txt', 'wb') as f:
    #     np.savetxt(f, Qp)
    # with open('Qr.txt', 'wb') as f:
    #     np.savetxt(f, Qr)
    # with open('Qv.txt', 'wb') as f:
    #     np.savetxt(f, Qv)
    # with open('li.txt', 'wb') as f:
    #     np.savetxt(f, li)
    # with open('lj.txt', 'wb') as f:
    #     np.savetxt(f, lj)

    minnm = n
    if m < minnm:
        minnm = m
    N = N + 1
    M = M + 1
    m = m + 1
    n = n + 1
    nedges = nedges + 1
    q = np.zeros(N, float)
    mi = np.zeros(Qp[N - 1], int)
    mj = np.zeros(Qp[N - 1], int)
    print("Qp[N]", Qp[N - 1])
    medges = 1

    lwork1 = np.ones(m, int)
    lind1 = -1 * np.ones(m, int)
    lwork2 = np.ones(n, int)
    lind2 = -1 * np.ones(n, int)

    max_col_nonzeros = 0
    for j in range(1, N):
        col_nonzeros = Qp[j + 1] - Qp[j]
        if col_nonzeros >= max_col_nonzeros:
            max_col_nonzeros = col_nonzeros
    print("max_col_nonzeros", max_col_nonzeros)
    se1 = np.zeros(max_col_nonzeros + 1, int)
    se2 = np.zeros(max_col_nonzeros + 1, int)
    sw = np.zeros(max_col_nonzeros + 1, float)
    sqi = np.zeros(max_col_nonzeros + 1, int)
    sqj = np.zeros(max_col_nonzeros + 1, int)

    for j in range(1, N):
        smalledges = 1
        nsmall1 = 1
        nsmall2 = 1
        for nzi in range(Qp[j], Qp[j + 1]):
            print("nzi", nzi)
            i = Qr[nzi]
            v1 = li[i]
            v2 = lj[i]
            sv1 = -1
            sv2 = -1
            if lind1[v1] < 0:
                sv1 = nsmall1
                lind1[v1] = sv1
                lwork1[sv1] = v1
                nsmall1 += 1
            else:
                sv1 = lind1[v1]
            if lind2[v2] < 0:
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
        nsmall1 -= 1
        nsmall2 -= 1
        smalledges -= 1

        if smalledges == 0:
            q[j] = 0
            continue
        res, mi1 = intmatch(nsmall1, nsmall2, smalledges, se1, se2, sw)
        q[j] = res
        smi = mi1
        for k in range(1, smalledges + 1):
            if smi[k] > 0:
                mi[medges] = sqi[k]
                mj[medges] = sqj[k]
                medges += 1

        for k in range(1, nsmall1 + 1):
            lind1[lwork1[k]] = -1

        for k in range(1, nsmall2 + 1):
            lind2[lwork2[k]] = -1
    medges -= 1
    return q, mi, mj, medges


if __name__ == "__main__":
    # M = 6
    # N = 5
    M = 12
    N = 12
    nedges = 31
    li = np.array([0, 1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 6])
    lj = np.array([0, 1, 2, 3, 4, 1, 2, 1, 3, 1, 4, 5, 2])
    m = 6
    n = 5
    # Qp = np.array([0, 6, 8, 10, 11, 12, 5, 7, 9, 5, 7, 9, 5, 7,
    #             9, 2, 3, 4, 12, 1, 2, 3, 4, 1, 2, 3, 4, 1, 1, 1, 5])
    # Qr = np.array([0, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4,
    #              5, 5, 5, 5, 6, 7, 7, 7, 8, 9, 9, 9, 10, 11, 12, 12])
    Qv = np.array(
        [0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
         0.5, 0.5,
         0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    )
    Qp = np.array([-1, 0, 5, 8, 11, 14, 18, 19, 22, 23, 26, 27, 28, 30])
    Qr = np.array([-1, 5, 7, 9, 10, 11, 4, 6, 8, 4, 6, 8, 4, 6, 8,
                   1, 2, 3, 11, 0, 1, 2, 3, 0, 1, 2, 3, 0, 0, 0, 4])
    Qp = Qp + 1
    Qr = Qr + 1
    print(column_maxmatchsum(M, N, Qp, Qr, Qv, m, n, nedges, li, lj))

# 12 12 6 5 31
# [ 0  1  6  9 12 15 19 20 23 24 27 28 29 31]
# (14,)
# [ 0  6  8 10 11 12  5  7  9  5  7  9  5  7  9  2  3  4 12  1  2  3  4  1
#   2  3  4  1  1  1  5]
# (31,)
# [0.  0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5
#  0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5]
# (31,)
# [0 1 1 1 1 2 2 3 3 4 4 5 6]
# (13,)
# [0 1 2 3 4 1 2 1 3 1 4 5 2]
# (13,)


# [0. , 2. , 0.5, 0.5, 0.5, 1. , 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
# [ 0,  6,  8, 10, 11,  5,  5,  5,  3, 12,  1,  2,  1,  2,  1,  1,  1, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]
# [ 0,  1,  1,  1,  1,  2,  3,  4,  5,  5,  6,  7,  8,  9, 10, 11, 12, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]
# 16
