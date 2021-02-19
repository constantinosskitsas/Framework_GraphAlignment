import scipy.sparse as sps
import numpy as np
from ..LREA.bipartiteMatching import bipartite_matching_setup, bipartite_matching_primal_dual, edge_list


# def print(*args):
#     pass


def othermaxplus(dim, li, lj, lw, m, n):

    if dim == 1:
        i1 = lj
        i2 = li
        N = n
    else:
        i1 = li
        i2 = lj
        N = m

    dimmax1 = np.zeros(N)
    dimmax2 = np.zeros(N)
    dimmaxind = np.zeros(N)
    nedges = len(li)

    for i in range(nedges):
        if lw[i] > dimmax2[i1[i]]:
            if lw[i] > dimmax1[i1[i]]:
                dimmax2[i1[i]] = dimmax1[i1[i]]
                dimmax1[i1[i]] = lw[i]
                dimmaxind[i1[i]] = i2[i]
            else:
                dimmax2[i1[i]] = lw[i]

    omp = np.zeros(len(lw))
    for i in range(nedges):
        if i2[i] == dimmaxind[i1[i]]:
            omp[i] = dimmax2[i1[i]]
        else:
            omp[i] = dimmax1[i1[i]]

    return omp


def othersum(si, sj, s, m, n):
    rowsum = accumarray(si, s, m)
    return rowsum[si] - s


def accumarray(xij, xw, n):
    sums = np.zeros(n)

    for i in range(len(xij)):
        sums[xij[i]] += xw[i]

    return sums


def round_messages(messages, S, w, alpha, beta, rp, ci, tripi, n, m, perm):
    print("rm")
    # ai=np.zeros(length(tripi),1);
    # ai(tripi>0)=messages(perm);
    # %disp(ai)
    ai = np.zeros(len(ci))
    ai[:len(messages)] = messages
    print(rp)
    print(ci)
    print(ai)
    # val ma mb mi
    m, n, val, noute, match1 = bipartite_matching_primal_dual(
        rp, ci, ai, tripi, m, n)
    ma, mb = edge_list(m, n, val, noute, match1)
    print(ma)
    print(mb)
    # print(mi)
    # matchweight = sum(w[mi])
    # cardinality = sum(mi)
    # overlap = (mi*(S*mi))/2
    # f = alpha*matchweight + beta*overlap
    # return f, matchweight, cardinality, overlap
    print("VAL:", val)
    return [val, ma, mb]


def main(S, li, lj, a=1, b=1, gamma=0.99, dtype=2, maxiter=100, verbose=1):
    li = np.array(li, int)
    li -= 1
    lj = np.array(lj, int)
    lj -= 1
    S = sps.csr_matrix(S)
    w = np.ones(len(li))

    nedges = len(li)
    nsquares = S.count_nonzero() // 2
    m = max(li) + 1
    n = max(lj) + 1

    # compute a vector that allows us to transpose data between squares.
    sui, suj, _ = sps.find(sps.triu(S, 1))
    SI = sps.csr_matrix(
        (list(range(1, len(sui)+1)), (sui, suj)), shape=S.shape
    )
    SI = SI + SI.transpose()
    si, sj, sind = sps.find(SI)
    sind = [x-1 for x in sind]
    SP = sps.csr_matrix(
        ([1]*len(si), (si, sind)), shape=(S.shape[0], nsquares)
    )
    sij, sijrs, _ = sps.find(SP)
    sind = list(range(SP.count_nonzero()))
    spair = sind[:]
    spair[::2] = sind[1::2]
    spair[1::2] = sind[::2]

    # Initialize the messages
    ma = np.zeros(nedges, int)
    mb = np.zeros(nedges, int)
    ms = np.zeros(S.count_nonzero())
    sums = np.zeros(nedges)

    damping = gamma
    curdamp = 1
    alpha = a
    beta = b

    # Initialize history
    # hista = np.zeros((maxiter, 4))
    # histb = np.zeros((maxiter, 4))
    fbest = 0
    fbestiter = 0
    print(li)
    print(lj)
    DA = sps.csc_matrix((w, (li+1, lj+1)))
    rp, ci, ai, tripi, matn, matm = bipartite_matching_setup(
        DA, None, None, None)

    # rp, ci, ai, tripi, matn, matm = bipartite_matching_setupc(w, li, lj, m, n)
    mperm = [x for x in tripi if x > 0]

    for it in range(maxiter):
        print(it)
        prevma = ma
        prevmb = mb
        prevms = ms
        prevsums = sums
        curdamp = damping * curdamp

        omaxb = np.array(
            [max(0, x) for x in othermaxplus(2, li, lj, mb, m, n)],
        )
        omaxa = np.array(
            [max(0, x)for x in othermaxplus(1, li, lj, ma, m, n)],
        )

        msflip = ms[spair]
        mymsflip = msflip+beta
        mymsflip = [min(beta, x) for x in mymsflip]
        mymsflip = [max(0, x) for x in mymsflip]

        sums = accumarray(sij, mymsflip, nedges)

        ma = alpha*w - omaxb + sums
        mb = alpha*w - omaxa + sums

        ms = alpha*w[sij]-(omaxb[sij] + omaxa[sij])
        ms += othersum(sij, sijrs, mymsflip, nedges, nsquares)

        if dtype == 1:
            ma = curdamp*(ma) + (1-curdamp)*(prevma)
            mb = curdamp*(mb) + (1-curdamp)*(prevmb)
            ms = curdamp*(ms) + (1-curdamp)*(prevms)
        elif dtype == 2:
            ma = ma + (1-curdamp)*(prevma+prevmb-alpha*w+prevsums)
            mb = mb + (1-curdamp)*(prevmb+prevma-alpha*w+prevsums)
            ms = ms + (1-curdamp)*(prevms+prevms[spair]-beta)
        elif dtype == 3:
            ma = curdamp*ma + (1-curdamp)*(prevma+prevmb-alpha*w+prevsums)
            mb = curdamp*mb + (1-curdamp)*(prevmb+prevma-alpha*w+prevsums)
            ms = curdamp*ms + (1-curdamp)*(prevms+prevms[spair]-beta)

        print(ma)
        print(mb)
        print(ms)
        # f, matchweight, cardinality, overlap
        hista = round_messages(ma, S, w, alpha, beta, rp,
                               ci, tripi, matn, matm, mperm)
        histb = round_messages(mb, S, w, alpha, beta, rp,
                               ci, tripi, matn, matm, mperm)
        if hista[0] > fbest:
            fbestiter = iter
            # mbest = ma
            mbest = hista[1:]
            fbest = hista[0]
        if histb[0] > fbest:
            fbestiter = -iter
            # mbest = mb
            mbest = mbest = histb[1:]
            fbest = histb[0]
        # return
    return mbest
