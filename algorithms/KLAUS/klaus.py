import scipy.sparse as sps
import numpy as np
from ..bipartiteMatching import bipartite_matching_setup, bipartite_matching_primal_dual, edge_list, matching_indicator, bipartite_matching
from ..maxrowmatchcpp import column_maxmatchsum


def maxrowmatch_mock(Q, li, lj, m, n):
    # print(Q)  # everything -1
    q = np.array([
        2.0000, 0.5000, 0.5000, 0.5000, 1.0000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000
    ])
    data = [
        [6, 1, 1],
        [8, 1, 1],
        [10, 1, 1],
        [11, 1, 1],
        [5, 2, 1],
        [5, 3, 1],
        [5, 4, 1],
        [3, 5, 1],
        [12, 5, 1],
        [1, 6, 1],
        [2, 7, 1],
        [1, 8, 1],
        [2, 9, 1],
        [1, 10, 1],
        [1, 11, 1],
        [1, 12, 1]
    ]
    nli = np.array([d[0] for d in data], int)
    nlj = np.array([d[1] for d in data], int)
    nw = np.array([d[2] for d in data])
    # print(nli)
    # print(nlj)
    # print(nw)
    SM = sps.csr_matrix((nw, (nli, nlj)))
    return q, SM


def to_matlab(array, diff=1):
    res = array.copy()
    res += diff
    return np.insert(res, [0], [0])


def to_python(array, diff=1):
    res = array.copy()
    res -= diff
    return np.delete(res, [0])


def maxrowmatch(Q, li, lj, m, n):
    Qt = Q.T
    # print(Qt.shape)
    # print(Qt.nnz)
    # print(li.shape)
    q, mi, mj = column_maxmatchsum(
        # li.shape[0],
        # lj.shape[0],
        Qt.shape[1],
        Qt.shape[0],
        Qt.indptr,
        Qt.indices,
        Qt.data,
        m,
        n,
        # Qt.nnz,
        li.shape[0],
        li,
        lj
    )
    # print(q)
    # print(mi)
    # print(mj)

    SM = sps.csr_matrix((np.ones(mi.shape[0]), (mi, mj)), shape=Q.shape)
    return q, SM


def debugm(U):
    uu = np.array(sps.find(U), float).T
    debug(uu[:, uu[1].argsort()])


def debug(arr):
    print("###")
    print(arr[:51])
    print(arr[-50:])
    print("$$$")


def main(S, w, li, lj, a=1, b=1, gamma=0.4, stepm=25, rtype=1, maxiter=1000, verbose=True):
    m = max(li) + 1
    n = max(lj) + 1

    rp, ci, ai, tripi, _, _ = bipartite_matching_setup(
        None, to_matlab(li), to_matlab(lj), to_matlab(w, 0), m, n)

    mperm = tripi[tripi > 0]

    S = sps.csr_matrix(S, dtype=float)
    U = sps.csr_matrix(S.shape)

    xbest = np.zeros(len(w))

    flower = 0.0
    fupper = np.inf
    next_reduction_iteration = stepm

    if verbose:
        print('{:5s}   {:4s}   {:8s}   {:7s} {:7s} {:7s}  {:7s} {:7s} {:7s} {:7s}'.format(
            'best', 'iter', 'norm-u', 'lower', 'upper', 'cur', 'obj', 'weight', 'card', 'overlap'))

    for it in range(1, maxiter+1):
        # print((b/2)*S + U-U.T)
        q, SM = maxrowmatch((b/2)*S + U-U.T, li, lj, m, n)
        # print(q)
        # print(SM.A)
        # print(SM.shape)
        # return
        x = a*w + q
        ai = np.zeros(len(tripi))
        ai[tripi > 0] = x[mperm-1]
        _, _, val, noute, match1 = bipartite_matching_primal_dual(
            rp, ci, ai, tripi, m+1, n+1)

        # print(rp)
        # print(ci)
        # print(ai)
        # print(tripi)
        # print(val)
        # print(noute)
        # print(match1)

        mi = matching_indicator(rp, ci, match1, tripi, m, n)[1:]

        # ma, mb = edge_list(m+1, n+1, val, noute, match1)
        # matchval = np.dot(mi, w)
        # overlap = np.dot(mi, S*mi/2)
        # card = len(ma)
        # f = a*matchval + b*overlap

        matchval = sum(w[mi > 0])
        card = sum(mi)

        overlap = np.dot(mi.T, (S*mi))/2
        f = a*matchval + b*overlap

        # print(ma)
        # print(mb)
        # print(mi)
        # print(f)
        # print(val)

        # return

        if val < fupper:
            fupper = val
            next_reduction_iteration = it+stepm
        if f > flower:
            flower = f
            itermark = '*'
            xbest = mi
            # matching = (ma, mb)
        else:
            itermark = ' '

        if rtype == 1:
            pass
        elif rtype == 2:
            mw = S*x  # remove first 0 from x?
            mw = a*w + b/2*mw

            ai = np.zeros(len(tripi))
            ai[mperm2] = mw[mperm1]
            _, _, val, noute, match1 = bipartite_matching_primal_dual(
                rp, ci, ai, tripi, m+1, n+1)

            mx = matching_indicator(rp, ci, match1, tripi, m, n)[1:]
            ma, mb = edge_list(m, n, val, noute, match1)

            matchval = np.dot(mx, w)
            # overlap = np.dot(mx, S*mx/2)
            overlap = np.dot(mi.T, (S*mi))/2
            card = len(ma)
            f = a*matchval + b*overlap

            if f > flower:
                flower = f
                itermark = '**'
                mi = mx
                xbest = mw
                # matching = (ma, mb)

        if verbose:
            print('{:5s}   {:4d}   {:8.1e}   {:5.2f} {:5.2f} {:5.2f}  {:5.2f} {:5.2f} {:5.2f} {:5.2f}'.format(
                itermark, it, np.linalg.norm(U.nnz),
                flower, fupper, val,
                f, matchval, card, overlap
            ))

        if it == next_reduction_iteration:
            gamma = gamma*0.5
        if gamma < 1e-24:
            break
        next_reduction_iteration = it+stepm

        if (fupper-flower) < 1e-2:
            break

        # print((gamma*mi))
        # print(sps.diags(gamma*mi))
        # print(sps.triu(SM))
        GM = sps.diags(gamma*mi)
        # print(GM)
        # print()
        # print(GM * sps.triu(SM))
        # print()
        # print(sps.tril(SM).T * GM)
        # print()

        U = U - GM * sps.triu(SM) + sps.tril(SM).T * GM
        # print(U)

        U.data = U.data.clip(-0.5, 0.5)

        debugm(U)

    m, n, val, noute, match1 = bipartite_matching(
        None, to_matlab(li), to_matlab(lj), to_matlab(xbest, 0))
    ma, mb = edge_list(m, n, val, noute, match1)

    return ma, mb

    # return np.array(matching)
