from . import ex, quadtree
from algorithms import bipartitewrapper as bmw
import numpy as np
import scipy
try:
    import lapjv
except:
    pass


def colmax(matrix):
    ma = np.arange(matrix.shape[0])
    # mb = matrix.argmax(1).A1
    mb = matrix.argmax(1).flatten()
    return ma, mb


def colmin(matrix):
    ma = np.arange(matrix.shape[0])
    # mb = matrix.argmin(1).A1
    mb = matrix.argmin(1).flatten()
    return ma, mb


def superfast(l2, asc=True):
    # print(f"superfast: init")
    # l2 = l2.A
    n = l2.shape[0]
    ma = np.zeros(n, int)
    mb = np.zeros(n, int)
    rows = set()
    cols = set()
    vals = np.argsort(l2, axis=None)
    vals = vals if asc else vals[::-1]

    # i = 0
    # for x, y in zip(*np.unravel_index(vals, l2.shape)):
    for val in vals:
        x, y = np.unravel_index(val, l2.shape)
        if x in rows or y in cols:
            continue
        # print(f"superfast: {i}/{n}")
        # i += 1
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
            n = sim.shape[0]
            if (n & (n-1) == 0) and n != 0:
                _log.debug("binary! speeding up..")
                return quadtree.superfast_binbin(sim)
            else:
                return superfast(sim, asc=False)
        elif mt == 3:
            # _sim = -sim.A
            _sim = -sim
            try:
                return jv(_sim)
            except Exception:
                return scipy.optimize.linear_sum_assignment(_sim)
        elif mt == 30:
            # _sim = -np.log(sim.A)
            _sim = -np.log(sim)
            try:
                return jv(_sim)
            except Exception:
                return scipy.optimize.linear_sum_assignment(_sim)
        elif mt == 98:
            return scipy.sparse.csgraph.min_weight_full_bipartite_matching(-sim)
        elif mt == 99:
            return bmw.getmatchings(sim)

    if mt < 0:
        if cost is None:
            raise Exception("Empty cost matrix")
        if mt == -1:
            return colmin(cost)
        elif mt == -2:
            n = cost.shape[0]
            if (n & (n-1) == 0) and n != 0:
                _log.debug("binary! speeding up..")
                return quadtree.superfast_binbin(-cost)
            else:
                return superfast(cost)
        elif mt == -3:
            # _cost = cost.A
            _cost = cost
            try:
                return jv(_cost)
            except Exception:
                return scipy.optimize.linear_sum_assignment(_cost)
        elif mt == -30:
            # _cost = np.log(cost.A)
            _cost = np.log(cost)
            try:
                return jv(_cost)
            except Exception:
                return scipy.optimize.linear_sum_assignment(_cost)
        elif mt == -98:
            return scipy.sparse.csgraph.min_weight_full_bipartite_matching(cost)
        elif mt == -99:
            return bmw.getmatchings(-cost)

    raise Exception("wrong matching config")


# @profile
# def init():
#     # size = 35000
#     size = 1000
#     # sim = np.arange((size*size)).reshape(size, -1)
#     sim = np.ones((size*size)).reshape(size, -1)
#     # sim = scipy.sparse.csr_matrix(sim)
#     return sim


# if __name__ == "__main__":

#     sim = init()
#     print(sim.shape)
#     print(sim.size)
#     # while(1):
#     #     pass
#     # superfast(sim)
#     # superfast(sim, asc=False)
#     colmax(sim)
