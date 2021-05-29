import numpy as np
import scipy.sparse as sps
from algorithms.isorank.isorank2 import create_L


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
    # alpha = 0.95
    A = normout_rowstochastic(A).T.astype(dtype)
    B = normout_rowstochastic(B).T.astype(dtype)
    Zvecs = Zvecs.astype(dtype=dtype)
    Wvecs = Wvecs.astype(dtype=dtype)
    nB = np.shape(B)[0]
    nA = np.shape(A)[0]
    # A and B are now row stochastic, so no need for A'x or B'x anywhere
    # operations needed are only Ax or Bx
    # GlobalSim = np.zeros((nA, nB), dtype=dtype)
    Sim = np.zeros((nA, nB), dtype=dtype)
    # GlobalSim = np.zeros((nA * nB), dtype=dtype)
    for i in range(np.shape(Zvecs)[1]):
        print(f"<{i}>")
        z = Zvecs[:, i]
        w = Wvecs[:, i]
        z = z / sum(z)
        w = w / sum(w)
        # Z = np.zeros((nA, iters + 1), dtype=dtype)  # A
        # W = np.zeros((nB, iters + 1), dtype=dtype)  # B
        Z = np.zeros((iters + 1, nA), dtype=dtype)  # A
        W = np.zeros((iters + 1, nB), dtype=dtype)  # B
        # Sim = np.zeros((nA, nB), dtype=dtype)
        # Sim = np.zeros((nA * nB), dtype=dtype)

        # W[:, 0] = w
        # Z[:, 0] = z
        W[0] = w
        Z[0] = z

        # print("dots")
        for k in range(1, iters + 1):
            # print(k)
            # Z[:, k] = np.dot(A, Z[:, k-1])
            # W[:, k] = np.dot(B, W[:, k-1])
            # Z[k] = np.dot(A, Z[k-1])
            # W[k] = np.dot(B, W[k-1])
            np.dot(A, Z[k-1], out=Z[k])
            np.dot(B, W[k-1], out=W[k])

        factor = 1.0 - alpha
        # print("krons")

        for k in range(iters):
            # print(k)
            # Sim += np.kron(Z[:, k] * alpha ** k, W[:, k]).reshape(nA, nB)
            # Sim += np.kron(Z[k] * factor, W[k]).reshape(nA, nB)
            # Sim += np.dot(
            #     (Z[k] * factor).reshape(nA, 1), W[k].reshape(1, nB)
            # )
            for i, w in enumerate(W[k]):
                Sim[:, i] += Z[k] * factor * w
            factor *= alpha
        # Sim += np.dot(
        #     (Z[iters] * alpha ** iters).reshape(nA, 1), W[iters].reshape(1, nB)
        # )
        for i, w in enumerate(W[iters]):
            Sim[:, i] += Z[iters] * factor * w
        # for k in range(iters):
        #     Z[k] *= factor
        #     factor *= alpha
        # Z[iters] *= alpha ** iters
        # Sim += np.dot(Z.T, W)

        # print("end")
        # Sim += np.kron(Z[:, iters] * alpha ** iters,
        #                W[:, iters]).reshape(nA, nB)
        # Sim += np.kron(Z[iters], W[iters]).reshape(nA, nB)
        # Sim *= (1 - alpha)
        # Sim *= alpha ** iters
        # Sim += np.dot(
        #     (Z[iters] * alpha ** iters).reshape(nA, 1), W[iters].reshape(1, nB)
        # )
        # GlobalSim += Sim
    # return GlobalSim
    return Sim
    # return GlobalSim.reshape(nA, nB)


# def main(A, B, alpha, iters):
def main(data, alpha, iters):

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
