from algorithms import regal, eigenalign, conealign, netalign, NSD, klaus, gwl, isorank, grasp, isorank2, bipartitewrapper as bmw
from data import similarities_preprocess
from sacred import Experiment
import numpy as np
import scipy.sparse as sps
from scipy.io import loadmat
import inspect
import matplotlib.pyplot as plt
from data import ReadFile
import pandas as pd
from math import log2
import sys
import os

ex = Experiment("experiment")


def plot(cx, filename):
    connects = np.zeros(cx.shape)
    for row, col in zip(cx.row, cx.col):
        connects[row, col] += 1
    plt.imshow(connects)
    plt.savefig(f'results/{filename}.png')
    plt.close('all')


def colmax(matrix):
    ma = np.arange(matrix.shape[0])
    mb = matrix.argmax(1).A1
    return ma, mb


def colmin(matrix):
    ma = np.arange(matrix.shape[0])
    mb = matrix.argmin(1).A1
    return ma, mb


def fast3(l2):
    num = np.shape(l2)[0]
    ma = np.zeros(num, int)
    mb = np.zeros(num, int)
    for _ in range(num):
        hi = np.where(l2 == np.amax(l2))
        hia = hi[0][0]
        hib = hi[1][0]
        ma[hia] = hia
        mb[hia] = hib
        l2[:, hib] = -np.inf
        l2[hia, :] = -np.inf
    return ma, mb


def fast4(l2):
    num = np.shape(l2)[0]
    ma = np.zeros(num)
    mb = np.zeros(num)
    for _ in range(num):
        hi = np.where(l2 == np.amin(l2))
        hia = hi[0][0]
        hib = hi[1][0]
        ma[hia] = hia
        mb[hia] = hib
        l2[:, hib] = np.inf
        l2[hia, :] = np.inf
    return ma, mb


def eval_align(ma, mb, gmb):

    try:
        gmab = np.arange(gmb.size)
        gmab[ma] = mb
        gacc = np.mean(gmb == gmab)

        mab = gmb[ma]
        acc = np.mean(mb == mab)

    except Exception as e:
        mab = np.zeros(mb.size, int) - 1
        gacc = acc = -1.0
    alignment = np.array([ma, mb, mab]).T
    alignment = alignment[alignment[:, 0].argsort()]
    return gacc, acc, alignment


def S3(A, B, ma, mb):
    A1 = np.sum(A, 1)
    B1 = np.sum(B, 1)
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
    res = Ce/div
    return res


def ICorS3GT(A, B, ma, mb, gmb, IC):
    A1 = np.sum(A, 1)
    B1 = np.sum(B, 1)
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


# def ics3(Tar, Src, Gt, IC=(True, True, True)):


def get_counterpart(alignment_matrix):
    counterpart_dict = {}

    if not sps.issparse(alignment_matrix):
        sorted_indices = np.argsort(alignment_matrix)

    n_nodes = alignment_matrix.shape[0]
    for node_index in range(n_nodes):

        if sps.issparse(alignment_matrix):
            row, possible_alignments, possible_values = sps.find(
                alignment_matrix[node_index])
            node_sorted_indices = possible_alignments[possible_values.argsort(
            )]
        else:
            node_sorted_indices = sorted_indices[node_index]
        counterpart = node_sorted_indices[-1]
        counterpart_dict[node_index] = counterpart
    return counterpart_dict


def score_MNC(alignment_matrix, adj1, adj2, countera, counterb):
    mnc = 0
    # print(adj1.data.tolist())
    # print(adj1.tolist())
    if sps.issparse(alignment_matrix):
        alignment_matrix = alignment_matrix.toarray()
    if sps.issparse(adj1):
        adj1 = adj1.toarray()
    if sps.issparse(adj2):
        adj2 = adj2.toarray()
    # counter_dict = get_counterpart(alignment_matrix)
    node_num = alignment_matrix.shape[0]
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
            except:
                pass
            # one_hop_neighbor_counter.append(counterb[count])

        num_stable_neighbor = np.intersect1d(
            new_one_hop_neighbor, np.array(one_hop_neighbor_counter)).shape[0]
        union_align = np.union1d(new_one_hop_neighbor, np.array(
            one_hop_neighbor_counter)).shape[0]

        sim = float(num_stable_neighbor) / union_align
        mnc += sim

    mnc /= node_num
    return mnc


@ex.capture
def evall(ma, mb, matrix, Tar, Src, Gt, eval_type=0, alg=np.random.rand(), verbose=True):
    sys.stdout = sys.__stdout__
    np.set_printoptions(threshold=100)
    # np.set_printoptions(threshold=np.inf)

    gmb, gmb1 = Gt
    gmb = np.array(gmb, int)
    gmb1 = np.array(gmb1, int)
    ma = np.array(ma, int)
    mb = np.array(mb, int)

    assert ma.size == mb.size
    res = np.array([
        eval_align(ma, mb, gmb),
        eval_align(mb, ma, gmb),
        eval_align(ma, mb, gmb1),
        eval_align(mb, ma, gmb1),
    ], dtype=object)

    accs = res[:, 0]
    best = np.argmax(accs)

    if max(accs) < 0:
        if eval_type:
            prefix = "#"
        else:
            # print("misleading evaluation")
            prefix = "!"
    elif eval_type and eval_type != best:
        # print("eval_type mismatch")
        prefix = "%"
    else:
        prefix = ""

    acc, acc2, alignment = res[eval_type]
    acc3 = S3(Tar.A, Src.A, ma, mb)
    acc4 = ICorS3GT(Tar.A, Src.A, ma, mb, gmb, True)
    acc5 = ICorS3GT(Tar.A, Src.A, ma, mb, gmb, False)

    mnc = np.array([
        score_MNC(matrix, Tar, Src, ma, mb),
        score_MNC(matrix, Tar, Src, mb, ma),
        score_MNC(matrix, Src, Tar, ma, mb),
        score_MNC(matrix, Src, Tar, mb, ma)
    ])

    acc6 = mnc[eval_type]

    print(f"{' ' + alg +' ':#^25}")
    # print(alignment, end="\n\n")
    print(res[:, :2].astype(float))
    print(mnc, " MNC from refina")
    print(f"{'#':#^25}")

    with open(f'results/{prefix}{alg}_{best}_.txt', 'wb') as f:
        np.savetxt(f, res[:, :2], fmt='%2.3f')
        np.savetxt(f, [mnc], fmt='%2.3f')
        np.savetxt(f, [["ma", "mb", "gmab"]], fmt='%5s')
        np.savetxt(f, alignment, fmt='%5d')

    if not verbose:
        sys.stdout = open(os.devnull, 'w')

    return acc, acc2, acc3, acc4, acc5, acc6


def e_to_G(e):
    n = np.amax(e) + 1
    nedges = e.shape[0]
    G = sps.csr_matrix((np.ones(nedges), e.T), shape=(n, n), dtype=int)
    G += G.T
    G.data = G.data.clip(0, 1)
    return G


def preprocess(Src, Tar, lalpha=1, mind=0.00001):
    L = similarities_preprocess.create_L(Tar, Src, alpha=lalpha, mind=mind)
    # L = similarities_preprocess.create_L(Src, Tar, alpha=lalpha, mind=mind)
    S = similarities_preprocess.create_S(Tar, Src, L)
    # S = similarities_preprocess.create_S(Src, Tar, L)
    li, lj, w = sps.find(L)

    return L, S, li, lj, w


@ex.config
def global_config():
    noise_level = 1
    edges = 1
    verbose = True
    maxiter = 100

    _preprocess = False
    lalpha = 1

    target = "data/arenas_orig.txt"
    source = f"data/noise_level_{noise_level}/edges_{edges}.txt"
    grand_truth = f"data/noise_level_{noise_level}/gt_{edges}.txt"

    Tar_e = np.loadtxt(target, int)
    Src_e = np.loadtxt(source, int)
    gt_e = np.loadtxt(grand_truth, int).T

    Tar = e_to_G(Tar_e)
    Src = e_to_G(Src_e)

    Gt = (
        gt_e[:, gt_e[1].argsort()][0],  # source -> target
        gt_e[:, gt_e[0].argsort()][1]   # target -> source
    )

    if _preprocess:
        L, S, li, lj, w = preprocess(Tar, Src, lalpha)
        # L, S, li, lj, w = preprocess(Src, Tar, lalpha)
    else:
        L = S = li = lj = w = np.empty(1)
        # try:
        #     L = sps.load_npz(f"data/L_{noise_level}_{edges}_{lalpha}.npz")
        #     S = sps.load_npz(f"data/S_{noise_level}_{edges}_{lalpha}.npz")
        # except:
        #     try:
        #         L = sps.load_npz(f"data/L_1_1_full.npz")
        #         S = sps.load_npz(f"data/S_1_1_full.npz")
        #     except:
        #         L = sps.load_npz(f"data/L_1_1_5.npz")
        #         S = sps.load_npz(f"data/S_1_1_5.npz")
        # li, lj, w = sps.find(L)

    _lim = args = arg = None
    dat = {
        val: None for val in ['A', 'B', 'S', 'L', 'w', 'lw', 'li', 'lj']
    }


@ex.named_config
def prep():
    _preprocess = True


@ex.named_config
def demo():
    _preprocess = False
    args = 1
    _lim = 100
    maxiter = 10
    lalpha = 1

    Src_e = np.loadtxt("data/arenas_orig.txt", int)
    # Src_e = np.loadtxt("data/noise_level_5/edges_1.txt", int)
    Src_e = Src_e[np.where(Src_e < _lim, True, False).all(axis=1)]

    Gt = np.random.RandomState(seed=55).permutation(_lim)

    Tar_e = Gt[Src_e]
    Gt = (
        Gt,
        Gt
    )

    Tar = e_to_G(Tar_e)
    Src = e_to_G(Src_e)

    # Src = Tar.copy()
    # Gt = np.arange(_lim)
    arg = [
        [None],
        [_lim * _lim],
        [_lim * _lim, 0.0],
        [_lim * _lim, None],
        [_lim/2/log2(_lim)],
        [_lim/2/log2(_lim), 0.0],
        [_lim/2/log2(_lim), None],
        [lalpha],
        [lalpha, 0.0],
        [lalpha, None],
    ][args]

    L, S, li, lj, w = preprocess(Src, Tar, *arg)

    # L, S, li, lj, w = preprocess(Src, Tar, None)
    # L, S, li, lj, w = preprocess(Src, Tar, 1)
    # L, S, li, lj, w = preprocess(Src, Tar, 1, mind=None)
    # L, S, li, lj, w = preprocess(Src, Tar, _lim/2/log2(_lim))
    # L, S, li, lj, w = preprocess(Src, Tar, _lim/2/log2(_lim), mind=None)
    # L, S, li, lj, w = preprocess(Src, Tar, 999)
    # L, S, li, lj, w = preprocess(Src, Tar, 999, mind=None)
    # print(L.A)
    # L = sps.csr_matrix(np.ones((Src.shape[0], Tar.shape[0])))
    # S = similarities_preprocess.create_S(Src, Tar, L)
    # li, lj, w = sps.find(L)


@ex.named_config
def load():

    _preprocess = False

    dat = "data/example-overlap.mat"
    # dat = "data/example-overlapv2.mat"
    # dat = "data/lcsh2wiki-small.mat"

    dat = {
        k: v for k, v in loadmat(dat).items() if k in {'A', 'B', 'S', 'L', 'w', 'lw', 'li', 'lj'}
    }

    Src = dat['A'] if 'A' in dat else None
    Tar = dat['B'] if 'B' in dat else None
    S = dat['S']
    L = dat['L']
    w = dat['w'] if 'w' in dat else dat['lw']
    w = w.flatten()
    li = dat['li'].flatten() - 1
    lj = dat['lj'].flatten() - 1

    L = sps.csr_matrix(np.ones((Src.shape[0], Tar.shape[0])))
    S = similarities_preprocess.create_S(Src, Tar, L)
    li, lj, w = sps.find(L)

    # gt_e = np.loadtxt("data/lcsh_gt.txt", int) - 1

    # Gt = np.zeros(Src.shape[0], int) - 1
    # Gt[gt_e[:, 0]] = gt_e[:, 1]


@ex.capture
def eval_regal(Tar, Src):

    REGAL_args = {
        'attributes': None,
        'attrvals': 2,
        'dimensions': 128,
        'k': 10,
        'untillayer': 2,
        'alpha': 0.01,
        'gammastruc': 1.0,
        'gammaattr': 1.0,
        'numtop': 10,
        'buckets': 2
    }

    matrix = regal.main(Tar, Src, REGAL_args)

    ma, mb = colmax(matrix)
    # ma, mb = bmw.getmatchings(matrix)
    # ma, mb = fast3(matrix.A)

    return evall(ma, mb, matrix,
                 alg=inspect.currentframe().f_code.co_name)


@ex.capture
def eval_eigenalign(Tar, Src):

    LREA_args = {
        'iters': 8,
        'method': "lowrank_svd_union",
        'bmatch': 3,
        'default_params': True
    }

    matrix = eigenalign.main(Tar.A, Src.A, **LREA_args)

    # ma, mb = colmax(matrix)
    # ma, mb = fast3(matrix.A)
    ma, mb = bmw.getmatchings(matrix)

    return evall(ma, mb, matrix,
                 alg=inspect.currentframe().f_code.co_name)


@ex.capture
def eval_conealign(Tar, Src):

    CONE_args = {
        'dim': 64,
        'window': 10,
        'negative': 1.0,
        'niter_init': 10,
        'reg_init': 1.0,
        'nepoch': 5,
        'niter_align': 10,
        'reg_align': 0.05,
        'bsz': 10,
        'lr': 1.0,
        'embsim': 'euclidean',
        'alignmethod': 'greedy',
        'numtop': 10
    }

    matrix = conealign.main(Tar.A, Src.A, CONE_args)

    ma, mb = colmax(matrix)
    # ma, mb = fast3(matrix.A)
    # ma, mb = bmw.getmatchings(matrix)

    return evall(ma, mb, matrix,
                 alg=inspect.currentframe().f_code.co_name)


@ex.capture
def eval_NSD(Tar, Src):

    NSD_args = {
        'alpha': 0.5,
        'iters': 10
    }

    matrix = NSD.run(Tar.A, Src.A, **NSD_args)

    # ma, mb = colmax(sps.csr_matrix(matrix))
    ma, mb = fast3(matrix)
    # ma, mb = bmw.getmatchings(matrix)

    return evall(ma, mb, matrix,
                 alg=inspect.currentframe().f_code.co_name)


@ex.capture
def eval_grasp(Tar, Src):

    GRASP_args = {
        'laa': 2,
        'icp': False,
        'icp_its': 3,
        'q': 100,
        'k': 20,
        'n_eig': Src.shape[0],
        # 'n_eig': 1131,
        'lower_t': 1.0,
        'upper_t': 50.0,
        'linsteps': True,
        'base_align': True
    }

    cost = grasp.main(Tar.A, Src.A, GRASP_args)
    # cost = grasp.main(Tar, Src, GRASP_args)
    # cost = grasp.main(Src.A, Tar.A, GRASP_args)
    # cost = grasp.main(Src, Tar, GRASP_args)

    # ma, mb = colmin(sps.csr_matrix(cost))
    ma, mb = fast4(cost)
    # ma, mb = bmw.getmatchings(cost)

    return evall(ma, mb, cost,
                 alg=inspect.currentframe().f_code.co_name)


@ex.capture
def eval_gwl(Tar, Src):

    GW_args = {
        'opt_dict': {
            'epochs': 1,            # the more u study the worse the grade man
            'batch_size': 100000,   # should be all data I guess?
            'use_cuda': False,
            'strategy': 'soft',
            'beta': 1e-1,
            'outer_iteration': 400,
            'inner_iteration': 1,
            'sgd_iteration': 300,
            'prior': False,
            'prefix': 'results',
            'display': False
        },
        'hyperpara_dict': {
            'dimension': 100,
            'loss_type': 'L2',
            'cost_type': 'cosine',
            'ot_method': 'proximal'
        }
    }

    matrix, cost = gwl.main(Tar, Src, **GW_args)

    # ma, mb = colmin(sps.csr_matrix(cost))
    # ma, mb = fast4(cost)

    # ma, mb = colmax(sps.csr_matrix(matrix))
    ma, mb = fast3(matrix)
    # ma, mb = bmw.getmatchings(matrix)

    return evall(ma, mb, matrix,
                 alg=inspect.currentframe().f_code.co_name)


@ex.capture
def eval_isorank(Tar, Src, L, S, w, li, lj):

    ISO_args = {
        # 'L' = L,
        'alpha': 0.5,
        'tol': 1e-12,
        'maxiter': 1,
        'verbose': True
    }

    matrix = isorank2.main(Tar.A, Src.A, **ISO_args)

    # ma, mb = colmax(sps.csr_matrix(matrix))
    ma, mb = fast3(matrix)
    # ma, mb = bmw.getmatchings(matrix)

    return evall(ma, mb, matrix,
                 alg=inspect.currentframe().f_code.co_name)


@ex.capture
def eval_netalign(S, w, li, lj, maxiter):

    NET_args = {
        'a': 1,
        'b': 1,
        'gamma': 0.99,
        'dtype': 2,
        'maxiter': maxiter,
        'verbose': True
    }

    matrix = netalign.main(S, w, li, lj, **NET_args)

    # ma, mb = colmax(matrix)
    # ma, mb = fast3(matrix.A)
    ma, mb = bmw.getmatchings(matrix)

    return evall(ma, mb, matrix,
                 alg=inspect.currentframe().f_code.co_name)


@ex.capture
def eval_klaus(S, w, li, lj, maxiter):

    KLAU_args = {
        'a': 1,
        'b': 1,
        'gamma': 0.4,
        'stepm': 25,
        'rtype': 1,
        'maxiter': maxiter,
        'verbose': True
    }

    matrix = klaus.main(S, w, li, lj, **KLAU_args)

    # ma, mb = colmax(matrix)
    # ma, mb = fast3(matrix.A)
    ma, mb = bmw.getmatchings(matrix)

    return evall(ma, mb, matrix,
                 alg=inspect.currentframe().f_code.co_name)


@ex.automain
def main(verbose, Gt, Tar, Src, S, L, w, li, lj, noise_level, edges, lalpha, gt_e, Src_e, Tar_e):
    print()

    # try:
    #     plot(sps.coo_matrix(Src), "Src")
    #     plot(sps.coo_matrix(Tar), "Tar")
    #     plot(sps.coo_matrix(L), "L")
    #     # plot(sps.coo_matrix(S), "S")
    # except:
    #     pass

    # sps.save_npz(f"S_{noise_level}_{edges}_{lalpha}.npz", S)
    # sps.save_npz(f"L_{noise_level}_{edges}_{lalpha}.npz", L)

    if not verbose:
        sys.stdout = open(os.devnull, 'w')

    # with np.printoptions(threshold=np.inf) as a:
    with np.printoptions(threshold=100) as a:
        print(np.array(list(enumerate(Gt[0]))))

    results = np.array([
        eval_gwl(),             # non-stable
        eval_grasp(),
        eval_conealign(),
        eval_regal(),           # non-stable

        eval_eigenalign(),
        eval_NSD(),
        eval_isorank(),

        eval_netalign(),
        eval_klaus(),
    ])

    # df = pd.DataFrame(results)
    # df.to_csv(f'results/exp_{noise_level}_{edges}.csv', index=False)

    sys.stdout = sys.__stdout__
    print("\n")
    print(results)
