import numpy as np
from numpy import inf, nan
import scipy.sparse as sps
import scipy

# from algorithms import bipartiteMatching
# from algorithms.NSD.NSD import fast2, findnz1
# from data import similarities_preprocess, ReadFile
# from evaluation import evaluation
# from evaluation.evaluation import check_with_identity
# from evaluation.evaluation_design import remove_edges_directed
# from experiment.similarities_preprocess import create_L
from math import floor, log2


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
            d = 1 - abs(a[i]-b[j]) / max(a[i], b[j])
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

# def main(A, B, L=None, alpha=0.5, tol=1e-12, maxiter=1, verbose=True):


def main(data, alpha=0.5, tol=1e-12, maxiter=1, verbose=True, lalpha=None, weighted=True):

    dtype = np.float32
    # dtype = np.float64

    # Src = data['Src'].A
    # Tar = data['Tar'].A
    # L = data['L'].A
    Src = data['Src']
    Tar = data['Tar']
    L = data['L']

    if lalpha is not None:
        L = create_L(Src, Tar, lalpha=lalpha,
                     weighted=weighted).A.astype(dtype)

    n1 = Tar.shape[0]
    n2 = Src.shape[0]

    # normalize the adjacency matrices
    d1 = 1 / Tar.sum(axis=1)
    d2 = 1 / Src.sum(axis=1)

    d1[d1 == inf] = 0
    d2[d2 == inf] = 0
    d1 = d1.reshape(-1, 1)
    d2 = d2.reshape(-1, 1)

    W1 = d1*Tar
    W2 = d2*Src

    W2aT = (alpha*W2.T).astype(dtype)
    K = ((1-alpha) * L).astype(dtype)
    W1 = W1.astype(dtype)

    S = np.ones((n2, n1), dtype=dtype) / (n1 * n2)  # Map target to source
    # IsoRank Algorithm in matrix form
    for it in range(1, maxiter + 1):
        # print(it)
        prev = S.flatten()
        if alpha is None:
            S = W2.T.dot(S).dot(W1)
        else:
            # S = (alpha*W2.T).dot(S).dot(W1) + (1-alpha) * L
            S = W2.T.dot(S).dot(W1) + K
            # W2aT.dot(S, out=S)
            # S.dot(W1, out=S)
            # S += K
        delta = np.linalg.norm(S.flatten()-prev, 2)
        if verbose:
            print("Iteration: ", it, " with delta = ", delta)
        if delta < tol:
            break

    return S


# if __name__ == "__main__":
#     data1 = "../../data/arenas_orig.txt"
#     data2 = "../../data/arenas_orig.txt"
#     #data2 = "../../data/noise_level_1/edges_1.txt"
#     gt = "../../data/noise_level_1/gt_1.txt"

#     # gma, gmb = ReadFile.gt1(gt)
#     G1 = ReadFile.edgelist_to_adjmatrix1(data1)
#     G2 = ReadFile.edgelist_to_adjmatrix1(data2)
#     G2 = remove_edges_directed(G2)
#     G1 = remove_edges_directed(G1)
#     gma, gmb = ReadFile.gt1(gt)
#     # adj = ReadFile.edgelist_to_adjmatrixR(data1, data2)
#     Ai, Aj = np.loadtxt(data1, int).T
#     n = max(max(Ai), max(Aj)) + 1
#     nedges = len(Ai)
#     Aw = np.ones(nedges)
#     A = sps.csr_matrix((Aw, (Ai, Aj)), shape=(n, n), dtype=int)
#     A = A + A.T

#     Bi, Bj = np.loadtxt(data2, int).T
#     m = max(max(Bi), max(Bj)) + 1
#     medges = len(Bi)
#     Bw = np.ones(medges)
#     B = sps.csr_matrix((Bw, (Bi, Bj)), shape=(m, m), dtype=int)
#     B = B + B.T
#     L = similarities_preprocess.create_L(A, B, alpha=18)
#     #S = similarities_preprocess.create_S(A, B, L)
#     # print(S)
#     #li, lj, w = sps.find(L)
#     a = 0
#     S = main(L, G1, G2, a, maxiter=100)
#     ma, mb = fast2(S)
#     print(np.amax(mb))
#     #S1 = main(L, G1, G2, 0.6, maxiter=1)
#     #ma1, mb1 = fast2(S1)
#     #S1= main(L, G1, G2, a)
#     #ma, mb = fast2(S1)
#     #acc = evaluation.accuracy(gma, gmb, mb, ma)
#     #acc1 = evaluation.accuracy(gma, gmb, mb1, ma1)
#     # Sa=sps.csr_matrix(S)
#     #nzi1, nzj1, nzv1 = findnz1(S1)
#     #DA = sps.csc_matrix((nzv1, (nzi1, nzj1)), shape=(len(nzi1), len(nzj1)))
#    # m, n, val, noute, match1 = (
#     # bipartiteMatching.bipartite_matching(DA, nzi1, nzj1, nzv1))
#     #ma, mb = bipartiteMatching.edge_list(m, n, val, noute, match1)
#     #acc = evaluation.accuracy(gma, gmb, mb, ma)
#     # print(ma,mb)
#     #print (acc)
#     #acc1 = evaluation.accuracy(gma, gmb, mb1, ma1)
#     acc1 = check_with_identity(mb+1)
#     print(mb)
#     # print(ma,mb)
#     print(acc1)
