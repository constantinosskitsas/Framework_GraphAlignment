from evaluation import evaluation
import networkx
from networkx.algorithms.bipartite.tests.test_matrix import sparse
import numpy as np
import scipy
import networkx.algorithms.bipartite.matching
from scipy.sparse.csgraph._matching import maximum_bipartite_matching

from algorithms import bipartiteMatching
from data import ReadFile
from data.ReadFile import nonzeroentries, edgelist_to_adjmatrix1
from scipy.sparse import csc_matrix, csr_matrix
scipy.sparse.csgraph


def normout_rowstochastic(P):
    n = np.shape(P)[0]
    colsums = sum(P, 1)-1
    pi, pj, pv = findnz_alt(P)
    pv = np.divide(pv, colsums[pi])
    ef = (colsums[pi])
    Q = csc_matrix((pv, (pi, pj)), shape=(n, n)).toarray()
    return Q


def makesparse(D):
    num = np.shape(D)[0]*0.2
    num = int(num)
    print(num)
    X = np.copy(D)
    V = np.sort(X, axis=0)
    X = np.sort(X)

    count = 0
    for i in range(np.shape(D)[0]):
        test = X[i, num]
        count = 0
        for j in range(np.shape(D)[1]):
            test1 = V[j, num]
            if (D[i, j] <= test and D[i, j] <= test1):
                D[i, j] = 0
                count = count+1
        print("metrw", count)
    Da = csc_matrix(D)
    return Da, D


def nsd(A, B, alpha, iters, Zvecs, Wvecs):
    A = normout_rowstochastic(A)
    B = normout_rowstochastic(B)
    nB = np.shape(B)[0]
    nA = np.shape(A)[0]
    # A and B are now row stochastic, so no need for A'x or B'x anywhere
    # operations needed are only Ax or Bx
    GlobalSim = np.zeros((nA, nB))
    for i in range(np.shape(Zvecs)[1]):
        z = Zvecs[:, i]
        w = Wvecs[:, i]
        z = z / sum(z)
        w = w / sum(w)
        Z = np.zeros((nA, iters + 1))  # A
        W = np.zeros((nB, iters + 1))  # B
        Sim = np.zeros((nA, nB))

        W[:, 0] = w
        Z[:, 0] = z
        print(B)
        for k in range(1, iters + 1):
            #W[:, k] = np.dot(B.conj(), W[:, k - 1])
            W[:, k] = np.dot(B.transpose(), W[:, k-1])
            #W[:, k] = np.dot(B.transpose(), W[:, k - 1])
            Z[:, k] = np.dot(A.transpose(), Z[:, k - 1])
        print("z")
        # print(W)
        for k in range(iters):
            test1 = pow(alpha, k)
            test2 = test1 * Z[:, k]
            sa = np.shape(test2)[0]
            test2 = test2.reshape((sa, 1))
            sa = np.shape(W[:, k].conjugate())[0]
            asd = W[:, k].conjugate().reshape((1, sa))
            test3 = np.dot(test2, asd)
            Sim = Sim + test3
        Sim = (1 - alpha) * Sim
        test1 = pow(alpha, iters)
        test2 = test1 * Z[:, iters]
        sa = np.shape(test2)[0]
        test2 = test2.reshape((sa, 1))
        sa = np.shape(W[:, iters].conjugate())[0]
        asd = W[:, iters].conjugate().reshape((1, sa))
        test3 = np.dot(test2, asd)
        Sim = Sim + test3
        GlobalSim = GlobalSim + Sim
    return GlobalSim


def findnz_alt(A):
    size = np.count_nonzero(A)
    a = np.zeros((size), dtype=int)
    b = np.zeros((size), dtype=int)
    c = np.zeros((size), dtype=float)
    count = 0
    for i in range(np.shape(A)[0]):
        for j in range(np.shape(A)[1]):
            if (A[i, j] == 1):
                a[count] = i
                b[count] = j
                c[count] = 1.0
                count = count+1
    return a, b, c


def findnz1(A):
    out_tpl = np.nonzero(A)
    out_arr = A[np.nonzero(A)]
    return out_tpl[0], out_tpl[1], out_arr


def findnz(A):
    rwcol = np.nonzero(A)
    print(np.shape(A))
    print(np.shape(rwcol))
    rw = rwcol[0]
    col = rwcol[1]
    print(np.shape(rw))
    print(np.shape(col))
    vi = np.zeros(np.shape(rw))
    for i in range(np.shape(rw)[0]):
        vi = A[rw, col]
    return rw, col, vi


def fast2(l2):
    num = np.shape(l2)[0]
    print(num)
    print(np.shape(l2))
    zero_els = np.count_nonzero(l2)
    print(zero_els, "hiii")
    ma = np.zeros(np.shape(l2)[0])
    mb = np.zeros(np.shape(l2)[0])
    i = 0
    while i < num:
        hi = (np.where(l2 == np.amax(l2)))
        mb[hi[1][0]] = hi[0][0]
        ma[hi[1][0]] = hi[1][0]
        print(hi[0][0])
        print(hi[1][0])
        print(l2[hi[0][0], hi[1][0]])
        l2[hi[0][0], :] = 0
        l2[:, hi[1][0]] = 0
        i = i + 1
        print(i)
    return ma, mb


# def main(A, B, alpha, iters):
def main(data, alpha, iters):

    Tar = data['Tar']
    Src = data['Src']

    print("hey1")
    X = nsd(
        Tar.A, Src.A, alpha, iters,
        np.ones((np.shape(Tar)[0], 1)),
        np.ones((np.shape(Src)[0], 1))
    )
    print(X)
    #np.savetxt("array.txt",X, fmt="%s")
    asa = (np.shape(X))[0]
    print("asa", asa)
    #nzi1 = np.zeros(asa, int)
    #nzj1 = np.zeros(asa, int)
    #nzv1 = np.zeros(asa, float)
    #nzi1, nzj1, nzv1 = findnz1(X)
    # DA = scipy.sparse.csc_matrix(
    #   (nzv1, (nzi1, nzj1)), shape=(asa, asa))
    print("hey4")
    # print(DA)
    # x1 = np.copy(X)
    # ma1, mb1 = fast2(x1)
    # newarr,pr=makesparse(X)
    #np.savetxt("array1.txt", pr, fmt="%s")
    print("hey5")
    # print(newarr)
    #nzi, nzj, nzv = findnz1(newarr)
    #asa = maximum_bipartite_matching(newarr,perm_type='column')

    #Ba = networkx.from_scipy_sparse_matrix(newarr)
    #ma = networkx.max_weight_matching(Ba)
    #mb = np.zeros(asa, float)
    # print(np.shape(DA),"hi")
    # m, n, val, noute, match1 = (
    # bipartiteMatching.bipartite_matching(newarr, nzi, nzj, nzv))
    #ma, mb = bipartiteMatching.edge_list(m, n, val, noute, match1)
    return X


if __name__ == "__main__":

    data1 = "../../data/arenas_orig.txt"
    data2 = "../../data/noise_level_1/edges_1.txt"
    gt = "../../data/noise_level_1/gt_1.txt"

    gma, gmb = ReadFile.gt1(gt)
    G1 = ReadFile.edgelist_to_adjmatrix1(data1)
    G2 = ReadFile.edgelist_to_adjmatrix1(data2)
    adj = ReadFile.edgelist_to_adjmatrixR(data1, data2)
    ma, mb = run(G1, G2)
    print(np.shape(ma))
    print(np.shape(mb))
    print(gma)
    print(gmb)
    print(ma)
    print(mb)
    acc = evaluation.accuracy(gma, gmb, mb, ma)
    print(acc, "acc")
