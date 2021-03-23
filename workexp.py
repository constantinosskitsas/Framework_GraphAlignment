from algorithms import regal, eigenalign, conealign, netalign, NSD, klaus, gwl, isorank, grasp, isorank2
from data import similarities_preprocess
from sacred import Experiment
import numpy as np
import scipy.sparse as sps
from scipy.io import loadmat
import inspect
import matplotlib.pyplot as plt

ex = Experiment("experiment")

# def print(*args):
#     pass


def plot(cx, filename):
    connects = np.zeros(cx.shape)
    for row, col in zip(cx.row, cx.col):
        connects[row, col] += 1
    plt.imshow(connects)
    plt.savefig(f'results/{filename}.png')
    plt.close('all')


def fast2(l2):
    num = np.shape(l2)[0]
    ma = np.zeros(num)
    mb = np.zeros(num)
    for _ in range(num):
        hi = np.where(l2 == np.amax(l2))
        hia = hi[1][0]
        hib = hi[0][0]
        ma[hia] = hia
        mb[hia] = hib
        l2[:, hia] = 0
        l2[hib, :] = 0
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


def evall(gmb, ma, mb, alg=np.random.rand(), eval_type=None):
    np.set_printoptions(threshold=100)
    # np.set_printoptions(threshold=np.inf)
    print(f"\n\n\n#### {alg} ####\n")
    gmb = np.array(gmb, int)
    ma = np.array(ma, int)
    mb = np.array(mb, int)

    assert ma.size == mb.size

    res = np.array([
        eval_align(ma, mb, gmb),
        eval_align(mb, ma, gmb),
        eval_align(ma-1, mb-1, gmb),
        eval_align(mb-1, ma-1, gmb)
    ], dtype=object)

    accs = res[:, 0]
    best = np.argmax(accs)

    if max(accs) < 0:
        if eval_type:
            best = eval_type
            prefix = "#"
        else:
            print("misleading evaluation")
            prefix = "!"
    elif eval_type and eval_type != best:
        print("eval_type mismatch")
        prefix = "%"
    else:
        prefix = ""

    acc, acc2, alignment = res[best]

    print(alignment, end="\n\n")
    print(res[:, :2])
    print("\n###############")

    with open(f'results/{prefix}{alg}_{best}_.txt', 'wb') as f:
        np.savetxt(f, res[:, :2], fmt='%2.2f', newline="\n\n")
        np.savetxt(f, [["ma", "mb", "gmab"]], fmt='%5s')
        np.savetxt(f, alignment, fmt='%5d')

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


def preprocess(Tar, Src, lalpha):
    L = similarities_preprocess.create_L(Tar, Src, alpha=lalpha)
    S = similarities_preprocess.create_S(Tar, Src, L)
    li, lj, w = sps.find(L)

    return L, S, li, lj, w


@ex.config
def global_config():
    noise_level = 1
    edges = 1
    maxiter = 100

    _preprocess = False
    lalpha = 15

    target = "data/arenas_orig.txt"
    source = f"data/noise_level_{noise_level}/edges_{edges}.txt"
    grand_truth = f"data/noise_level_{noise_level}/gt_{edges}.txt"

    Tar_e = np.loadtxt(target, int)
    Src_e = np.loadtxt(source, int)
    gt_e = np.loadtxt(grand_truth, int).T

    Tar = e_to_G(Tar_e)
    Src = e_to_G(Src_e)
    Gt = gt_e[:, gt_e[0].argsort()][1]

    if _preprocess:
        L, S, li, lj, w = preprocess(Tar, Src, lalpha)
    else:
        try:
            L = sps.load_npz(f"data/L_{noise_level}_{edges}_{lalpha}.npz")
            S = sps.load_npz(f"data/S_{noise_level}_{edges}_{lalpha}.npz")
        except:
            L = sps.load_npz(f"data/L_1_1_15.npz")
            S = sps.load_npz(f"data/S_1_1_15.npz")
        li, lj, w = sps.find(L)

    _lim = None
    dat = {
        val: None for val in ['A', 'B', 'S', 'L', 'w', 'lw', 'li', 'lj']
    }


@ex.named_config
def prep():
    _preprocess = True


@ex.named_config
def demo():
    _preprocess = False
    _lim = 100
    maxiter = 10
    # lalpha = 1/15
    lalpha = 100

    Src_e = np.loadtxt("data/arenas_orig.txt", int)
    # print(Src_e)
    # print(np.amax(Src_e))
    # Src_e = np.random.permutation(np.amax(Src_e)+1)[Src_e]
    # print(Src_e)
    # print(np.amax(Src_e))
    # Src_e = np.arange(np.amax(Src_e)+1)[Src_e]

    Src_e = Src_e[np.where(Src_e < _lim, True, False).all(axis=1)]
    # print(Src_e)
    # print(np.amax(Src_e))
    # print(Src_e.shape)
    Gt = np.random.permutation(_lim)
    Tar_e = Gt[Src_e]

    Tar = e_to_G(Tar_e)
    Src = e_to_G(Src_e)

    # Src = Tar.copy()
    # Gt = np.arange(_lim)

    # L, S, li, lj, w = preprocess(Tar, Src, lalpha)

    L = sps.csr_matrix(np.ones((Tar.shape[0], Src.shape[0])))
    S = similarities_preprocess.create_S(Tar, Src, L)
    li, lj, w = sps.find(L)


@ex.named_config
def load():

    _preprocess = False

    # dat = {
    #     k: v for k, v in loadmat("data/example-overlap.mat").items() if k in {'A', 'B', 'S', 'L', 'w', 'lw', 'li', 'lj'}
    # }

    dat = {
        k: v for k, v in loadmat("data/lcsh2wiki-small.mat").items() if k in {'A', 'B', 'S', 'L', 'w', 'lw', 'li', 'lj'}
    }

    Src = dat['A']
    Tar = dat['B']
    S = dat['S']
    L = dat['L']
    w = dat['w'] if 'w' in dat else dat['lw']
    w = w.flatten()
    li = dat['li'].flatten() - 1
    lj = dat['lj'].flatten() - 1

    gt_e = np.loadtxt("data/lcsh_gt.txt", int) - 1

    Gt = np.zeros(Src.shape[0], int) - 1
    Gt[gt_e[:, 0]] = gt_e[:, 1]


@ex.capture
def eval_regal(Gt, Tar, Src):
    # adj = G_to_Adj(Tar, Src)
    adj = G_to_Adj(Src, Tar)

    alignmatrix = regal.main(adj.A)
    ma = np.arange(alignmatrix.shape[0])
    mb = alignmatrix.argmax(1).A1

    return evall(Gt, ma, mb,
                 alg=inspect.currentframe().f_code.co_name, eval_type=0)


@ex.capture
def eval_eigenalign(Gt, Tar, Src):
    ma, mb, _, _ = eigenalign.main(Tar.A, Src.A, 8, "lowrank_svd_union", 3)

    return evall(Gt, ma, mb,
                 alg=inspect.currentframe().f_code.co_name, eval_type=3)


@ex.capture
def eval_conealign(Gt, Tar, Src):

    alignmatrix = conealign.main(Src.A, Tar.A)
    ma = np.arange(alignmatrix.shape[0])
    mb = alignmatrix.argmax(1).A1

    return evall(Gt, ma, mb,
                 alg=inspect.currentframe().f_code.co_name, eval_type=0)


@ex.capture
def eval_NSD(Gt, Tar, Src):

    ma, mb = NSD.run(Tar.A, Src.A)

    return evall(Gt, ma, mb,
                 alg=inspect.currentframe().f_code.co_name, eval_type=0)


@ex.capture
def eval_grasp(Gt, Tar, Src):

    ma, mb = grasp.main(Tar.A, Src.A, alg=2, base_align=True)
    # ma, mb = grasp.main(Tar, Src, alg=2, base_align=True)

    return evall(Gt, ma, mb,
                 alg=inspect.currentframe().f_code.co_name, eval_type=0)


@ex.capture
def eval_gwl(Gt, Tar, Src):

    opt_dict = {
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
    }

    ma, mb = gwl.main(Tar, Src, opt_dict)

    return evall(Gt, ma, mb,
                 alg=inspect.currentframe().f_code.co_name, eval_type=0)


@ex.capture
def eval_isorank(Gt, Tar, Src, L, S, w, li, lj, maxiter):

    # ma, mb = isorank.main(S, w, li, lj, a=0.2, b=0.8,
    #                       alpha=None, rtype=1, maxiter=maxiter)

    alignment_matrix = isorank2.main(Tar.A, Src.A, maxiter=1)
    ma, mb = fast2(alignment_matrix)

    return evall(Gt, ma, mb,
                 alg=inspect.currentframe().f_code.co_name, eval_type=0)


@ex.capture
def eval_netalign(Gt, S, w, li, lj, maxiter):

    ma, mb = netalign.main(S, w, li, lj, a=0, maxiter=maxiter)

    return evall(Gt, ma, mb,
                 alg=inspect.currentframe().f_code.co_name, eval_type=2)


@ex.capture
def eval_klaus(Gt, S, w, li, lj, maxiter):

    ma, mb = klaus.main(S, w, li, lj, a=0, maxiter=maxiter)

    return evall(Gt, ma, mb,
                 alg=inspect.currentframe().f_code.co_name, eval_type=3)


@ex.automain
def main(Gt, Tar, Src, S, L, w, li, lj, noise_level, edges, lalpha):
    # with np.printoptions(threshold=np.inf) as a:
    #     print(np.array(list(enumerate(Gt))))

    try:
        plot(sps.coo_matrix(Src), "Src")
        plot(sps.coo_matrix(Tar), "Tar")
        plot(sps.coo_matrix(L), "L")
        # plot(sps.coo_matrix(S), "S")
    except:
        pass

    # sps.save_npz(f"S_{noise_level}_{edges}_{lalpha}.npz", S)
    # sps.save_npz(f"L_{noise_level}_{edges}_{lalpha}.npz", L)

    results = np.array([
        # eval_regal(),
        # eval_eigenalign(),
        # eval_conealign(),
        # eval_NSD(),
        # eval_grasp(),
        # eval_gwl(),

        # eval_isorank(),
        eval_netalign(),

        # eval_klaus(),
    ])

    print("\n\n\n")
    print(results)
