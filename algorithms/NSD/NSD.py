import networkx
from networkx.algorithms.bipartite.tests.test_matrix import sparse
import numpy as np
import scipy
from algorithms.LREA import bipartiteMatching
from algorithms.LREA.bipartiteMatching import edge_list, bipartite_matching
from data.ReadFile import nonzeroentries, edgelist_to_adjmatrix1
from scipy.sparse import csc_matrix

def normout_rowstochastic(P, file):
    n = np.shape(P)[0]
    colsums = sum(P, 1)-1
    pi, pj, pv =findnz_alt(P)
    pv=np.divide(pv,colsums[pi])
    ef=(colsums[pi])
    Q = csc_matrix((pv, (pi, pj)), shape=(n, n)).toarray()
    return Q


def nsd(A, B, alpha, iters, Zvecs, Wvecs):
    A = normout_rowstochastic(A, "data/noise_level_1/arenas_orig.txt")
    B = normout_rowstochastic(B, "data/noise_level_1/edges_1.txt")
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
        #print(W)
        for k in range(iters):
            test1 = pow(alpha, k)
            test2 = test1 * Z[:, k]
            sa=np.shape(test2)[0]
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
    size=np.count_nonzero(A)
    a = np.zeros((size), dtype=int)
    b = np.zeros((size), dtype=int)
    c = np.zeros((size), dtype=float)
    count=0
    for i in range(np.shape(A)[0]):
        for j in range(np.shape(A)[1]):
            if (A[i,j]==1):
                a[count]=i
                b[count]=j
                c[count]=1.0
                count=count+1
    return a,b,c
def findnz1(A):
    out_tpl = np.nonzero(A)
    out_arr = A[np.nonzero(A)]
    return out_tpl[0],out_tpl[1],out_arr
def findnz(A):
    rwcol=np.nonzero(A)
    print(np.shape(A))
    print(np.shape(rwcol))
    rw=rwcol[0]
    col=rwcol[1]
    print(np.shape(rw))
    print(np.shape(col))
    vi=np.zeros(np.shape(rw))
    for i in range(np.shape(rw)[0]):
        vi=A[rw,col]
    print(np.shape(vi))
    return rw,col,vi
def run():
    print("hey1")
    A = edgelist_to_adjmatrix1("data/noise_level_1/arenas_orig.txt")
    B = edgelist_to_adjmatrix1("data/noise_level_1/edges_1.txt")
    X = nsd(A, B, 0.8, 10, np.ones((np.shape(A)[0],1)), np.ones((np.shape(A)[0],1)))
    print(X)
    asa=(np.shape(X))[0]
    nzi1 = np.zeros(asa, int)
    nzj1 = np.zeros(asa, int)
    nzv1 = np.zeros(asa, float)
    nzi1,nzj1,nzv1=findnz1(X)
    DA = scipy.sparse.csc_matrix(
        (nzv1, (nzi1, nzj1)), shape=(asa, asa))
    print("hey4")
    print(DA)
    Ba=networkx.from_scipy_sparse_matrix(DA)
    ma = networkx.max_weight_matching(Ba)
    mb = np.zeros(asa, float)
    #print(np.shape(DA),"hi")
    #m, n, val, noute, match1 = (
        #bipartiteMatching.bipartite_matching(DA, nzi1, nzj1, nzv1))
    #ma, mb = bipartiteMatching.edge_list(m, n, val, noute, match1)
    #networkx.algorithms.bipartite.matching.hopcroft_karp_matching(X)

    print(mb)
    return ma,mb

