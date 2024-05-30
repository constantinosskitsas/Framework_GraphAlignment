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
    # dtype = np.float16
    dtype = np.float32
    # dtype = np.float64

    A = normout_rowstochastic(A).T.astype(dtype)
    B = normout_rowstochastic(B).T.astype(dtype)
    Zvecs = Zvecs.astype(dtype=dtype)
    Wvecs = Wvecs.astype(dtype=dtype)
    nB = np.shape(B)[0]
    nA = np.shape(A)[0]

    Sim = np.zeros((nA, nB), dtype=dtype)
    for i in range(np.shape(Zvecs)[1]):
        print(f"<{i}>")
        z = Zvecs[:, i]
        w = Wvecs[:, i]
        z = z / sum(z)
        w = w / sum(w)
        Z = np.zeros((iters + 1, nA), dtype=dtype)  # A
        W = np.zeros((iters + 1, nB), dtype=dtype)  # B
        W[0] = w
        Z[0] = z

        print("dots")
        for k in range(1, iters + 1):
            print(k)
            np.dot(A, Z[k-1], out=Z[k])
            np.dot(B, W[k-1], out=W[k])

        factor = 1.0 - alpha
        print("krons")
        for k in range(iters + 1):
            print(k)

            if k == iters:
                W[iters] *= alpha ** iters
                # Z[iters] *= alpha ** iters
            else:
                W[k] *= factor
                # Z[k] *= factor
                factor *= alpha

            # Sim += np.kron(Z[k], W[k]).reshape(nA, nB)

            # Sim += np.dot(
            #     Z[k].reshape(-1, 1), W[k].reshape(1, -1)
            # )

            # for i, w in enumerate(W[k]):
            #     Sim[:, i] += Z[k] * w

            intervals = 4
            for i in range(intervals):
                start = i * nA // intervals
                end = (i+1) * nA // intervals
                Sim[:, start:end] += np.dot(
                    Z[k].reshape(-1, 1), W[k][start:end].reshape(1, -1)
                )

    return Sim


# def main(A, B, alpha, iters):
def main(data, alpha, iters):
    print("NSD")
    # Src = data['Src'].A
    # Tar = data['Tar'].A
    Src = data['Src']
    Tar = data['Tar']
    # L = create_L(Src, Tar, 99999).A

    X = nsd(
        Src, Tar, alpha, iters,
        np.ones((Src.shape[0], 1)),
        np.ones((Tar.shape[0], 1)),
        # np.ones(Src.shape),
        # np.ones(Tar.shape),
        # L,
        # L.T,
    )

    return X
