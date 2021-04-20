from math import log2, floor
import numpy as np
import scipy.sparse as sps


def e_to_G(e):
    n = np.amax(e) + 1
    nedges = e.shape[0]
    G = sps.csr_matrix((np.ones(nedges), e.T), shape=(n, n), dtype=int)
    G += G.T
    G.data = G.data.clip(0, 1)
    return G


def test1():
    A = sps.csr_matrix(
        [
            [0, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0]
        ]
    )

    B = sps.csr_matrix(
        [
            [0, 1, 1, 1, 1],
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0]
        ]
    )

    L = sps.csr_matrix(
        [
            [0.6, 0.9, 0.3, 0.1, 0.0],
            [0.9, 0.6, 0.0, 0.0, 0.0],
            [0.3, 0.0, 0.5, 0.0, 0.0],
            [0.1, 0.0, 0.0, 0.4, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.5],
            [0.0, 1.0, 0.0, 0.0, 0.0]
        ]
    )

    S = sps.csr_matrix(
        [
            [0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1],
            [0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        ]
    )

    assert np.array_equal(S.A, create_S(A, B, L).A)

    return A, B, L, S


def create_S(A, B, L):
    n = A.shape[0]
    m = B.shape[0]

    rpAB, ciAB = L.indptr, L.indices
    nedges = len(ciAB)

    Si = []
    Sj = []

    wv = np.full(m, -1)
    ri1 = 0
    for i in range(n):
        print(f'{i}/{n}')
        for ri1 in range(rpAB[i], rpAB[i+1]):
            wv[ciAB[ri1]] = ri1

        for ip in A[i].nonzero()[1]:
            if i == ip:
                continue
            # for jp in L[ip].nonzero()[1]:
            # print(ip)
            for ri2 in range(rpAB[ip], rpAB[ip+1]):
                jp = ciAB[ri2]
                for j in B[jp].nonzero()[1]:
                    if j == jp:
                        continue
                    if wv[j] >= 0:
                        Si.append(ri2)
                        Sj.append(wv[j])
        for ri1 in range(rpAB[i], rpAB[i+1]):
            wv[ciAB[ri1]] = -1

    return sps.csr_matrix(([1]*len(Si), (Sj, Si)), shape=(nedges, nedges), dtype=int)


def create_L(A, B, alpha=1, mind=0.00001):
    n = A.shape[0]
    m = B.shape[0]

    if alpha is None:
        return sps.csr_matrix(np.ones((n, m)))

    a = A.sum(1)
    b = B.sum(1)

    a_p = [(i, m[0, 0]) for i, m in enumerate(a)]
    a_p.sort(key=lambda x: x[1])

    b_p = [(i, m[0, 0]) for i, m in enumerate(b)]
    b_p.sort(key=lambda x: x[1])

    ab_m = [0] * n
    s = 0
    e = floor(alpha * log2(m))
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
            d = 1 - abs(a[i, 0]-b[j, 0]) / a[i, 0]
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


def test2():
    for i in range(2, 8):
        print(f"### {i} ###")

        n, m = np.random.randint(2**i, 2**(i+1), size=(1, 2))[0]
        A = sps.csr_matrix(np.random.randint(2, size=(n, n)), dtype=int)
        B = sps.csr_matrix(np.random.randint(2, size=(m, m)), dtype=int)
        L = create_L(A, B)
        S = create_S(A, B, L)

        print(A.A)
        print(B.A)
        print(L.A)
        print(S.A)
        # print(n, m)
        print(A.shape)
        print(B.shape)
        print(L.shape)
        print(S.shape)


def test3():
    _lim = 6

    Src_e = np.loadtxt("data/arenas_orig.txt", int)
    # perm = np.random.permutation(np.amax(Src_e)+1)
    # print(perm)
    # print(perm[Src_e])
    # Src_e = perm[Src_e]
    # print(Src_e)
    # print(np.random.permutation(np.amax(Src_e)+1)[Src_e])

    # Src_e = np.random.permutation(np.amax(Src_e)+1)[Src_e]
    # print(Src_e)

    # print(np.amax(Src_e))
    # Src_e = np.arange(np.amax(Src_e)+1)[Src_e]

    Src_e = Src_e[np.where(Src_e < _lim, True, False).all(axis=1)]
    # print(Src_e)
    # print(np.amax(Src_e))
    # print(Src_e.shape)
    Gt = np.random.permutation(_lim)
    Tar_e = Gt[Src_e]

    Tar = e_to_G(Tar_e)
    Src = e_to_G(Src_e)

    print(Tar.A)
    print(Tar.sum(1))
    print(Src.A)
    print(Src.sum(1))
    # print(create_L(Src, Tar, 1).A)
    print(create_L(Tar, Src, 1, None).A)


if __name__ == "__main__":
    A, B, L, S = test1()
    # # # test2()

    # _lim = 5
    # Src_e = np.loadtxt("data/arenas_orig.txt", int)
    # Src_e = Src_e[np.where(Src_e < _lim, True, False).all(axis=1)]
    # Gt = np.random.permutation(_lim)
    # Tar_e = Gt[Src_e]

    # Tar = e_to_G(Tar_e)
    # Src = e_to_G(Src_e)

    # # L = create_L(Src, Tar)
    # # S = create_S(Src, Tar, L)

    # print(create_S(Src, Tar, create_L(Src, Tar)).A)
    # print(create_S(Tar, Src, create_L(Tar, Src)).A)

    # # print(A.A)
    # # print(B.A)
    # # print(create_L(A, B, 1).A)
    # # print(create_L(B, A, 1).A)
