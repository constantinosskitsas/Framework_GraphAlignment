import scipy.sparse as sps
import numpy as np
from ..LREA.bipartiteMatching import bipartite_matching_setup, bipartite_matching_primal_dual, edge_list


def main(S, w, li, lj, a=1, b=1, gamma=0.4, stepm=25, rtype=1, maxiter=1, verbose=1):
    m = max(li) + 1
    n = max(lj) + 1

    nzi = li.copy()
    nzi = np.insert(nzi, [0], [0])

    nzj = lj.copy()
    nzj = np.insert(nzj, [0], [0])

    ww = np.insert(w, [0], [0])

    rp, ci, ai, tripi, _, _ = bipartite_matching_setup(
        None, nzi, nzj, ww, m, n)

    mperm1 = [x-1 for x in tripi if x > 0]
    mperm2 = [i for i, x in enumerate(tripi) if x > 0]

    S = sps.csr_matrix(S, dtype=float)
    U = sps.csr_matrix(S.shape)

    # # xbest = w
    # # xbest(: ) = 0

    flower = 0
    fupper = np.inf
    next_reduction_iteration = stepm

    for it in range(maxiter):
        print(it)
