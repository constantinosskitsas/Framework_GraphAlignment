import numpy as np
import scipy.sparse as sps
from algorithms.isorank.isorank2 import create_L
#original code from https://github.com/nassarhuda/NetworkAlignment.jl/blob/master/src/NSD.jl

def normout_rowstochastic(P):
    n = P.shape[0]
    # colsums = sum(P, 1)-1
    colsums = np.sum(P, axis=0)
    # pi, pj, pv = findnz_alt(P)
    pi, pj, pv = sps.find(P)
    pv = np.divide(pv, colsums[pi])
    Q = sps.csc_matrix((pv, (pi, pj)), shape=(n, n)).toarray()
    return Q


def nsd(A, B, alpha, iters, Zvecs, Wvecs):
    dtype = np.float32
    A = normout_rowstochastic(A).T.astype(dtype)
    B = normout_rowstochastic(B).T.astype(dtype)
    Zvecs = Zvecs.astype(dtype=dtype)
    Wvecs = Wvecs.astype(dtype=dtype)
    nB = np.shape(B)[0]
    nA = np.shape(A)[0]

    Sim = np.zeros((nA, nB), dtype=dtype)
    for i in range(np.shape(Zvecs)[1]):
        z = Zvecs[:, i]
        w = Wvecs[:, i]
        z = z / sum(z)
        w = w / sum(w)
        Z = np.zeros((iters + 1, nA), dtype=dtype)  # A
        W = np.zeros((iters + 1, nB), dtype=dtype)  # B
        W[0] = w
        Z[0] = z
        for k in range(1, iters + 1):
            np.dot(A, Z[k-1], out=Z[k])
            np.dot(B, W[k-1], out=W[k])

        factor = 1.0 - alpha
        for k in range(iters + 1):
            if k == iters:
                W[iters] *= alpha ** iters
            else:
                W[k] *= factor
                factor *= alpha
            intervals = 4
            for i in range(intervals):
                start = i * nA // intervals
                end = (i+1) * nA // intervals
                Sim[:, start:end] += np.dot(
                    Z[k].reshape(-1, 1), W[k][start:end].reshape(1, -1)
                )

    return Sim

def main(data, alpha, iters):
    print("NSD")
    Src = data['Src']
    Tar = data['Tar']

    X = nsd(
        Src, Tar, alpha, iters,
        np.ones((Src.shape[0], 1)),
        np.ones((Tar.shape[0], 1)),
    )

    return X
