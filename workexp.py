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
    ma = np.zeros(num)
    mb = np.zeros(num)
    for _ in range(num):
        hi = np.where(l2 == np.amax(l2))
        hia = hi[0][0]
        hib = hi[1][0]
        ma[hia] = hia
        mb[hia] = hib
        l2[:, hib] = -100
        l2[hia, :] = -100
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
        l2[:, hib] = 1000
        l2[hia, :] = 1000
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


@ex.capture
def evall(ma, mb, Gt, eval_type=0, alg=np.random.rand(), verbose=True):
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

    print(f"{' ' + alg +' ':#^25}")
    # print(alignment, end="\n\n")
    print(res[:, :2].astype(float))
    print(f"{'#':#^25}")

    with open(f'results/{prefix}{alg}_{best}_.txt', 'wb') as f:
        np.savetxt(f, res[:, :2], fmt='%2.2f', newline="\n\n")
        np.savetxt(f, [["ma", "mb", "gmab"]], fmt='%5s')
        np.savetxt(f, alignment, fmt='%5d')

    if not verbose:
        sys.stdout = open(os.devnull, 'w')

    return acc, acc2


def e_to_G(e):
    n = np.amax(e) + 1
    nedges = e.shape[0]
    G = sps.csr_matrix((np.ones(nedges), e.T), shape=(n, n), dtype=int)
    G += G.T
    G.data = G.data.clip(0, 1)
    return G


def G_to_Adj(G1, G2):
    adj1 = sps.kron([[1, 0], [0, 0]], G1)
    adj2 = sps.kron([[0, 0], [0, 1]], G2)
    adj = adj1 + adj2
    adj.data = adj.data.clip(0, 1)
    return adj


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
    adj = G_to_Adj(Tar, Src)

    matrix = regal.main(adj.A)

    ma, mb = colmax(matrix)
    #ma, mb = fast3(matrix.A)
    #ma, mb = bmw.getmatchings(matrix)

    return evall(ma, mb,
                 alg=inspect.currentframe().f_code.co_name)


@ex.capture
def eval_eigenalign(Tar, Src):
    matrix = eigenalign.main(Tar.A, Src.A, 8, "lowrank_svd_union", 3)

    # ma, mb = colmax(matrix)
    # ma, mb = fast3(matrix.A)
    ma, mb = bmw.getmatchings(matrix)

    return evall(ma, mb,
                 alg=inspect.currentframe().f_code.co_name)


@ex.capture
def eval_conealign(Tar, Src):

    matrix = conealign.main(Tar.A, Src.A)

    # ma, mb = colmax(matrix)
    # ma, mb = fast3(matrix.A)
    ma, mb = bmw.getmatchings(matrix)

    return evall(ma, mb,
                 alg=inspect.currentframe().f_code.co_name)


@ex.capture
def eval_NSD(Tar, Src):

    matrix = NSD.run(Tar.A, Src.A)
    # ma, mb = NSD.run(Src.A, Tar.A)

    # ma, mb = colmax(sps.csr_matrix(matrix))
    ma, mb = fast3(matrix)
    # ma, mb = bmw.getmatchings(matrix)

    return evall(ma, mb,
                 alg=inspect.currentframe().f_code.co_name)


@ex.capture
def eval_grasp(Tar, Src):

    ma, mb = grasp.main(Src.A, Tar.A, alg=2, base_align=True)
    # ma, mb = grasp.main(Src, Tar, alg=2, base_align=True)

    return evall(ma, mb,
                 alg=inspect.currentframe().f_code.co_name)


@ex.capture
def eval_gwl(Tar, Src):

    opt_dict = {
        'epochs': 18,            # the more u study the worse the grade man
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
    }

    matrix,cost = gwl.main(Tar, Src, opt_dict)

    #ma, mb = colmax(sps.csr_matrix(matrix))
    #ma, mb = colmin(sps.csr_matrix(cost))
    #ma, mb = fast3(matrix)
    ma, mb = fast4(cost)
    # ma, mb = bmw.getmatchings(matrix)

    return evall(ma, mb,
                 alg=inspect.currentframe().f_code.co_name)


@ex.capture
def eval_isorank(Tar, Src, L, S, w, li, lj, maxiter):

    # ma, mb = isorank.main(S, w, li, lj, a=0.2, b=0.8,
    #                       alpha=None, rtype=1, maxiter=maxiter)

    matrix = isorank2.main(Tar.A, Src.A, maxiter=1)

    # ma, mb = colmax(sps.csr_matrix(matrix))
    ma, mb = fast3(matrix)
    # ma, mb = bmw.getmatchings(matrix)

    return evall(ma, mb,
                 alg=inspect.currentframe().f_code.co_name)


@ex.capture
def eval_netalign(S, w, li, lj, maxiter):

    matrix = netalign.main(S, w, li, lj, a=1, b=1,
                           gamma=0.999, maxiter=maxiter)

    # ma, mb = colmax(matrix)
    # ma, mb = fast3(matrix.A)
    ma, mb = bmw.getmatchings(matrix)

    return evall(ma, mb,
                 alg=inspect.currentframe().f_code.co_name)


@ex.capture
def eval_klaus(S, w, li, lj, maxiter):

    matrix = klaus.main(S, w, li, lj, a=1, b=1, gamma=0.9,
                        maxiter=maxiter)

    # ma, mb = colmax(matrix)
    # ma, mb = fast3(matrix.A)
    ma, mb = bmw.getmatchings(matrix)

    return evall(ma, mb,
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
        #eval_regal(),           # non-stable
        #eval_eigenalign(),
        #eval_conealign(),
        #eval_NSD(),
        #eval_grasp(),
        eval_gwl(),             # non-stable

        #eval_isorank(),
        #eval_netalign(),
        #eval_klaus(),
    ])

    # df = pd.DataFrame(results)
    # df.to_csv(f'results/exp_{noise_level}_{edges}.csv', index=False)

    sys.stdout = sys.__stdout__
    print("\n")
    print(results)
