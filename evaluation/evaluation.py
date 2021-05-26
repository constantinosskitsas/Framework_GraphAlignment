from . import ex
import numpy as np
import scipy.sparse as sps
import os


def S3(A, B, ma, mb):
    A1 = np.sum(A, 0)
    B1 = np.sum(B, 0)
    EdA1 = np.sum(A1)
    EdB1 = np.sum(B1)
    Ce = 0
    source = 0
    target = 0
    res = 0
    for ai, bi in zip(ma, mb):
        source = A1[ai]
        target = B1[bi]
        if source == target:  # equality goes in either of the cases below, different case for...
            Ce = Ce+source
        elif source < target:
            Ce = Ce+source
        elif source > target:
            Ce = Ce+target
    div = EdA1+EdB1-Ce
    # print(EdA1)
    # print(EdB1)
    # print(Ce)
    res = Ce/div
    return res


def ICorS3GT(A, B, ma, mb, gmb, IC):
    A1 = np.sum(A, 0)
    B1 = np.sum(B, 0)
    EdA1 = np.sum(A1)
    EdB1 = np.sum(B1)

    Ce = 0
    source = 0
    target = 0
    res = 0
    for ai, bi in zip(ma, mb):
        if (gmb[ai] == bi):
            source = A1[ai]
            target = B1[bi]
            if source == target:  # equality goes in either of the cases below, different case for...
                Ce = Ce+source
            elif source < target:
                Ce = Ce+source
            elif source > target:
                Ce = Ce+target
    if IC == True:
        res = Ce/EdA1
    else:
        div = EdA1+EdB1-Ce
        res = Ce/div
    return res


def score_MNC(adj1, adj2, countera, counterb):
    try:
        mnc = 0
        # print(adj1.data.tolist())
        # print(adj1.tolist())
        # if sps.issparse(alignment_matrix):
        #     alignment_matrix = alignment_matrix.toarray()
        if sps.issparse(adj1):
            adj1 = adj1.toarray()
        if sps.issparse(adj2):
            adj2 = adj2.toarray()
        # counter_dict = get_counterpart(alignment_matrix)
        # node_num = alignment_matrix.shape[0]
        for cri, cbi in zip(countera, counterb):
            a = np.array(adj1[cri, :])
            # a = np.array(adj1[i, :])
            one_hop_neighbor = np.flatnonzero(a)
            b = np.array(adj2[cbi, :])
            # neighbor of counterpart
            new_one_hop_neighbor = np.flatnonzero(b)

            one_hop_neighbor_counter = []
            # print(one_hop_neighbor)

            for count in one_hop_neighbor:
                indx = np.where(count == countera)
                try:
                    one_hop_neighbor_counter.append(counterb[indx[0][0]])
                except Exception:
                    pass
                # one_hop_neighbor_counter.append(counterb[count])

            num_stable_neighbor = np.intersect1d(
                new_one_hop_neighbor, np.array(one_hop_neighbor_counter)).shape[0]
            union_align = np.union1d(new_one_hop_neighbor, np.array(
                one_hop_neighbor_counter)).shape[0]

            sim = float(num_stable_neighbor) / union_align
            mnc += sim

        return mnc / countera.size
    except Exception:
        return -1


def panos_MNC(adj1, adj2, ma, mb):
    # src_exp = adj1[ma][:, ma]
    src_exp = adj1
    src_act = adj2[mb][:, mb]

    good = 0
    total = 0

    for i in range(src_exp.shape[0]):
        for j in range(src_exp.shape[1]):
            if src_exp[i, j] == 1 or src_act[i, j] == 1:
                if src_exp[i, j] == src_act[i, j]:
                    good += 1
                total += 1
    # with np.printoptions(linewidth=1000, suppress=True, threshold=np.inf):
    #     print(adj2)
    #     print(adj1)
    #     print(mb)
    #     print(adj1[mb][:, mb])
    #     print(diff)
    #     print(np.mean(diff == 0))
    return good/total


def eval_align(ma, mb, gmb):

    try:
        gmab = np.arange(gmb.size)
        gmab[ma] = mb
        gacc = np.mean(gmb == gmab)

        mab = gmb[ma]
        acc = np.mean(mb == mab)

    except Exception:
        mab = np.zeros(mb.size, int) - 1
        gacc = acc = -1.0
    alignment = np.array([ma, mb, mab]).T
    alignment = alignment[alignment[:, 0].argsort()]
    return gacc, acc, alignment


# @profile
@ex.capture
def evall(ma, mb, Src, Tar, Gt, _log, _run, alg, accs, save=False, eval_type=0):

    gmb, gmb1 = Gt
    gmb = np.array(gmb, int)
    gmb1 = np.array(gmb1, int)

    ma = np.array(ma, int)
    mb = np.array(mb, int)

    assert ma.size == mb.size

    _log.debug("matched %s out of %s", mb.size, gmb.size)

    res = np.array([
        eval_align(ma, mb, gmb),
        eval_align(mb, ma, gmb),
        eval_align(ma, mb, gmb1),
        eval_align(mb, ma, gmb1),
    ], dtype=object)

    with np.printoptions(suppress=True, precision=4):
        _log.debug("\n%s", res[:, :2].astype(float))

    acc, accb, alignment = res[eval_type]

    _accs = []

    if 0 in accs:
        _accs.append(acc)
    if 1 in accs:
        _accs.append(S3(Src, Tar, ma, mb))
    if 2 in accs:
        _accs.append(ICorS3GT(Src, Tar, ma, mb, gmb, True))
    if 3 in accs:
        _accs.append(ICorS3GT(Src, Tar, ma, mb, gmb, False))
    if 4 in accs:
        _accs.append(score_MNC(Src, Tar, ma, mb))
    if 5 in accs:
        _accs.append(panos_MNC(Src, Tar, ma, mb))

    if save:
        output_path = f"runs/{_run._id}/alignments"

        os.makedirs(output_path, exist_ok=True)

        i = 0
        while os.path.exists(f"{output_path}/{alg}_{i}.txt"):
            i += 1

        with open(f"{output_path}/{alg}_{i}.txt", 'w') as f:
            np.savetxt(f, res[:, :2], fmt='%2.3f')
            np.savetxt(f, [_accs], fmt='%2.3f')
            np.savetxt(f, [["ma", "mb", "gmab"]], fmt='%5s')
            np.savetxt(f, alignment, fmt='%5d')

    return np.array(_accs)
