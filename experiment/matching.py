from . import ex
from algorithms import bipartitewrapper as bmw
import numpy as np
import scipy
try:
    import lapjv
except:
    pass


def colmax(matrix):
    ma = np.arange(matrix.shape[0])
    mb = matrix.argmax(1).A1
    return ma, mb


def colmin(matrix):
    ma = np.arange(matrix.shape[0])
    mb = matrix.argmin(1).A1
    return ma, mb


def superfast(l2, asc=True):
    # print(f"superfast: init")
    l2 = l2.A
    n = np.shape(l2)[0]
    ma = np.zeros(n, int)
    mb = np.zeros(n, int)
    rows = set()
    cols = set()
    vals = np.argsort(l2, axis=None)
    vals = vals if asc else vals[::-1]
    i = 0
    for x, y in zip(*np.unravel_index(vals, l2.shape)):
        if x in rows or y in cols:
            continue
        # print(f"superfast: {i}/{n}")
        i += 1
        ma[x] = x
        mb[x] = y

        rows.add(x)
        cols.add(y)
    return ma, mb


def jv(dist):
    # print('hungarian_matching: calculating distance matrix')

    # dist = sci.spatial.distance_matrix(G1_emb.T, G2_emb.T)
    n = dist.shape[0]
    # print(np.shape(dist))
    # print('hungarian_matching: calculating matching')
    cols, rows, _ = lapjv.lapjv(dist)
    # print(cols)
    # print(rows)
    matching = np.c_[rows, np.linspace(0, n-1, n).astype(int)]
    # print(matching)
    matching = matching[matching[:, 0].argsort()]
    # print(matching)
    return matching.astype(int).T


# @profile
@ex.capture
def getmatching(sim, cost, mt, _log):

    _log.debug("matching type: %s", mt)

    if mt > 0:
        if sim is None:
            raise Exception("Empty sim matrix")
        if mt == 1:
            return colmax(sim)
        elif mt == 2:
            return superfast(sim, asc=False)
        elif mt == 3:
            return scipy.optimize.linear_sum_assignment(sim.A, maximize=True)
        elif mt == 30:
            return scipy.optimize.linear_sum_assignment(np.log(sim.A), maximize=True)
        elif mt == 4:
            return jv(-sim.A)
        elif mt == 40:
            return jv(-np.log(sim.A))
        elif mt == 98:
            return scipy.sparse.csgraph.min_weight_full_bipartite_matching(sim, maximize=True)
        elif mt == 99:
            return bmw.getmatchings(sim)

    if mt < 0:
        if cost is None:
            raise Exception("Empty cost matrix")
        if mt == -1:
            return colmin(cost)
        elif mt == -2:
            return superfast(cost)
        elif mt == -3:
            return scipy.optimize.linear_sum_assignment(cost.A)
        elif mt == -30:
            return scipy.optimize.linear_sum_assignment(np.log(cost.A))
        elif mt == -4:
            return jv(cost.A)
        elif mt == -40:
            return jv(np.log(cost.A))
        elif mt == -98:
            return scipy.sparse.csgraph.min_weight_full_bipartite_matching(cost)
        elif mt == -99:
            return bmw.getmatchings(np.exp(-cost.A))

    raise Exception("wrong matching config")
