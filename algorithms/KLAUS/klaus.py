import scipy.sparse as sps
import numpy as np
from ..maxrowmatchcpp import column_maxmatchsum
from .. import bipartitewrapper as bmw


def maxrowmatch(Q, li, lj, m, n):

    Qt = Q

    q, mi, mj = column_maxmatchsum(
        Qt.shape[0],
        Qt.shape[1],
        Qt.indptr,
        Qt.indices,
        Qt.data,
        m,
        n,
        li.shape[0],
        li,
        lj,
    )

    SM = sps.csr_matrix((np.ones(mi.shape[0]), (mj, mi)), shape=Qt.shape)
    return q, SM


def main(data, a=1, b=1, gamma=0.4, stepm=25, rtype=1, maxiter=1000, verbose=True):

    S = data['S']
    li = data['li']
    lj = data['lj']
    w = data['w']

    setup, m, n = bmw.bipartite_setup(li, lj, w)

    S = sps.csr_matrix(S, dtype=float)
    U = sps.csr_matrix(S.shape)

    xbest = np.zeros(len(w))

    flower = 0.0
    fupper = np.inf
    next_reduction_iteration = stepm

    if verbose:
        print('{:5s}   {:>4s}   {:>8s}   {:>7s} {:>7s} {:>7s}  {:>7s} {:>7s} {:>7s} {:>7s}'.format(
            'best', 'iter', 'norm-u', 'lower', 'upper', 'cur', 'obj', 'weight', 'card', 'overlap'))

    for it in range(1, maxiter+1):

        q, SM = maxrowmatch((b/2)*S + U-U.T, li, lj, m, n)

        x = a*w + q

        f, matchval, card, overlap, val, mi = bmw.round_messages(
            x, S, w, a, b, setup, m, n)

        if val < fupper:
            fupper = val
            next_reduction_iteration = it+stepm
        if f > flower:
            flower = f
            itermark = '*'
            xbest = mi
        else:
            itermark = ' '

        if rtype == 1:
            pass
        elif rtype == 2:

            mw = S*x
            mw = a*w + b/2*mw

            f, matchval, card, overlap, _, mx = bmw.round_messages(
                mw, S, w, a, b, setup, m, n)

            if f > flower:
                flower = f
                itermark = '**'
                mi = mx
                xbest = mw

        if verbose:
            print('{:5s}   {:4d}   {:8.1e}   {:7.2f} {:7.2f} {:7.2f}  {:7.2f} {:7.2f} {:7d} {:7d}'.format(
                itermark, it, np.linalg.norm(U.data, 1),
                flower, fupper, val,
                f, matchval, card, overlap
            ))

        if it == next_reduction_iteration:
            gamma = gamma*0.5
            if verbose:
                print(f'{"":5s}   {"":4s}   reducing step to {gamma}')
            if gamma < 1e-24:
                break
            next_reduction_iteration = it+stepm

        if (fupper-flower) < 1e-2:
            break

        GM = sps.diags(gamma*mi, format="csr")
        U = U - GM * sps.triu(SM) + sps.tril(SM).T * GM
        U.data = U.data.clip(-0.5, 0.5)

    return sps.csr_matrix((xbest, (li, lj)))
