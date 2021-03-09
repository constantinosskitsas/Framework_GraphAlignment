import scipy.sparse as sps
import numpy as np
from ..bipartiteMatching import bipartite_matching_setup, bipartite_matching_primal_dual, edge_list, matching_indicator
from ..maxrowmatch import column_maxmatchsum


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


def maxrowmatch(Q, nzi, nzj, m, n):
    Qt = Q.T
    Qp = Qt.indptr
    Qr = Qt.indices
    Qv = Qt.data

    q, mj, mi, medges = column_maxmatchsum(
        nzi.shape[0]-1,
        nzj.shape[0]-1,
        to_matlab(Qp),
        to_matlab(Qr),
        to_matlab(Qv, 0),
        m,
        n,
        Qv.shape[0]+1,
        nzi,
        nzj
    )
    mi = to_python(mi)[:medges]
    mj = to_python(mj)[:medges]
    q = to_python(q, 0)

    SM = sps.csr_matrix(([1]*medges, (mi, mj)), shape=Q.shape)
    return q, SM


def main(S, w, li, lj, a=1, b=1, gamma=0.4, stepm=25, rtype=1, maxiter=1000, verbose=1):
    m = max(li) + 1
    n = max(lj) + 1

    nzi = to_matlab(li)

    nzj = to_matlab(lj)

    ww = to_matlab(w, 0)

    rp, ci, ai, tripi, _, _ = bipartite_matching_setup(
        None, nzi, nzj, ww, m, n)

    mperm1 = [x-1 for x in tripi if x > 0]
    mperm2 = [i for i, x in enumerate(tripi) if x > 0]

    S = sps.csr_matrix(S, dtype=float)
    U = sps.csr_matrix(S.shape)

    xbest = np.zeros(len(w))

    flower = 0.0
    fupper = np.inf
    next_reduction_iteration = stepm

    matching = ()

    for it in range(maxiter):
        print(f"({it:03d}/{maxiter})")
        q, SM = maxrowmatch((b/2)*S + U-U.T, nzi, nzj, m, n)
        # print(q)
        # print(SM.A)
        # print(SM.shape)

        x = a*w + q
        ai = np.zeros(len(tripi))
        ai[mperm2] = x[mperm1]
        _, _, val, noute, match1 = bipartite_matching_primal_dual(
            rp, ci, ai, tripi, m+1, n+1)

        # print(rp)
        # print(ci)
        # print(ai)
        # print(tripi)
        # print(val)
        # print(noute)
        # print(match1)

        mi = matching_indicator(rp, ci, match1, tripi, m, n)
        mi = mi[1:]
        ma, mb = edge_list(m, n, val, noute, match1)

        matchval = np.dot(mi, w)
        overlap = np.dot(mi, S*mi/2)
        card = len(ma)
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
            xbest = mi
            matching = (ma, mb)

        if rtype == 1:
            pass
        elif rtype == 2:
            mw = S*x  # remove first 0 from x?
            mw = a*w + b/2*mw

            ai = np.zeros(len(tripi))
            ai[mperm2] = mw[mperm1]
            _, _, val, noute, match1 = bipartite_matching_primal_dual(
                rp, ci, ai, tripi, m+1, n+1)

            mx = matching_indicator(rp, ci, match1, tripi, m, n)
            mx = mx[1:]
            ma, mb = edge_list(m, n, val, noute, match1)

            matchval = np.dot(mx, w)
            overlap = np.dot(mx, S*mx/2)
            card = len(ma)
            f = a*matchval + b*overlap

            if f > flower:
                flower = f
                mi = mx
                xbest = mw
                matching = (ma, mb)

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

    return np.array(matching)
