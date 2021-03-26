from . import bipartiteMatching as bm
import numpy as np


# def debugm(U):
#     uu = np.array(sps.find(U), float).T
#     debug(uu[:, uu[1].argsort()])


# def debug(arr):
#     print("###")
#     print(arr[:51])
#     print(arr[-50:])
#     print("$$$")

def to_matlab(array, diff=1):
    res = array.copy()
    res += diff
    return np.insert(res, [0], [0])


def to_python(array, diff=1):
    res = array.copy()
    res -= diff
    return np.delete(res, [0])


def bipartite_setup(li, lj, w):
    m = max(li) + 1
    n = max(lj) + 1

    rp, ci, ai, tripi = bm.bipartite_matching_setup(
        None, to_matlab(li), to_matlab(lj), to_matlab(w, 0), m, n)[:4]

    mperm = tripi[tripi > 0]

    return (rp, ci, ai, tripi, mperm), m, n


def round_messages(messages, S, w, alpha, beta, setup, m, n):
    rp, ci, _, tripi, mperm = setup

    ai = np.zeros(len(tripi))
    ai[tripi > 0] = messages[mperm-1]
    _, _, val, _, match1 = bm.bipartite_matching_primal_dual(
        rp, ci, ai, tripi, m+1, n+1)

    mi = bm.matching_indicator(rp, ci, match1, tripi, m, n)[1:]

    matchweight = sum(w[mi > 0])
    cardinality = sum(mi)

    overlap = np.dot(mi.T, (S*mi))/2
    f = alpha*matchweight + beta*overlap

    return f, matchweight, cardinality, overlap, val, mi


def getmatchings(li, lj, xbest):
    m, n, val, noute, match1 = bm.bipartite_matching(
        None, to_matlab(li), to_matlab(lj), to_matlab(xbest, 0))
    ma, mb = bm.edge_list(m, n, val, noute, match1)

    return ma, mb
