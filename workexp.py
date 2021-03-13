from algorithms import regal, eigenalign, conealign, netalign, NSD, klaus, gwl, isorank, grasp
from data import similarities_preprocess
from sacred import Experiment
import numpy as np
import scipy.sparse as sps

ex = Experiment("experiment")

# def print(*args):
#     pass


def evall(gma, gmb, ma, mb):
    np.set_printoptions(threshold=100)
    # np.set_printoptions(threshold=np.inf)
    print("\n\n\n#### EVAL ####\n")
    gma = np.array(gma, int)
    gmb = np.array(gmb, int)
    ma = np.array(ma, int)
    mb = np.array(mb, int)
    # alignment = np.array([ma, mb], dtype=int).T
    # print(alignment)
    gmab = gmb[ma]
    print(np.array([ma, mb, gmab]).T)
    matches = np.sum(mb == gmab)
    acc = matches/len(gma)
    acc2 = matches/len(ma)
    print(f"\nacc({matches}/{len(gma)}): {acc}")
    if(len(gma) != len(ma)):
        print(f"acc2({matches}/{len(ma)}): {acc2}")
    print("\n")

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


@ex.config
def global_config():
    _lim = None
    maxiter = 100
    lalpha = 4

    noise_level = 5
    edges = 2

    data = f"data/noise_level_{noise_level}/edges_{edges}.txt"
    target = "data/arenas_orig.txt"
    gt = f"data/noise_level_{noise_level}/gt_{edges}.txt"

    gma, gmb = np.loadtxt(gt, int).T
    Ae = np.loadtxt(data, int)
    Be = np.loadtxt(target, int)


@ex.named_config
def demo():
    _lim = 200
    maxiter = 10
    lalpha = 2

    data = "data/arenas_orig.txt"

    Ae = np.loadtxt(data, int)
    # Ae = np.random.permutation(np.amax(Ae)+1)[Ae]

    gma = np.arange(_lim)
    gmb = np.random.permutation(_lim)
    Ae = Ae[np.where(Ae < _lim, True, False).all(axis=1)]
    Be = gmb[Ae]


def init():
    A = e_to_G(Ae)
    B = e_to_G(Be)
    L = similarities_preprocess.create_L(B, A, alpha=lalpha)
    S = similarities_preprocess.create_S(B, A, L)
    li, lj, w = sps.find(L)


@ex.capture
def eval_regal(Ae, Be, gma, gmb):
    adj = G_to_Adj(e_to_G(Ae), e_to_G(Be))

    alignmatrix = regal.main(adj.A)
    ma = np.arange(alignmatrix.shape[0])
    mb = alignmatrix.argmax(1).A1

    return evall(gma, gmb, ma, mb)


@ex.capture
def eval_eigenalign(Ae, Be, gma, gmb):
    G1 = e_to_G(Ae)
    G2 = e_to_G(Be)

    ma, mb, _, _ = eigenalign.main(G1.A, G2.A, 8, "lowrank_svd_union", 3)

    return evall(gma, gmb, ma-1, mb-1)


@ex.capture
def eval_conealign(Ae, Be, gma, gmb):
    G1 = e_to_G(Ae)
    G2 = e_to_G(Be)

    alignmatrix = conealign.main(G1.A, G2.A)
    ma = np.arange(alignmatrix.shape[0])
    mb = alignmatrix.argmax(1).A1

    return evall(gma, gmb, ma, mb)


@ex.capture
def eval_netalign(Ae, Be, gma, gmb, maxiter, lalpha):
    A = e_to_G(Ae)
    B = e_to_G(Be)
    L = similarities_preprocess.create_L(A, B, alpha=lalpha)
    S = similarities_preprocess.create_S(A, B, L)
    li, lj, w = sps.find(L)

    ma, mb = netalign.main(S, w, li, lj, a=0, maxiter=maxiter)

    return evall(gma, gmb, ma-1, mb-1)


@ex.capture
def eval_NSD(Ae, Be, gma, gmb):
    G1 = e_to_G(Ae)
    G2 = e_to_G(Be)

    ma, mb = NSD.run(G2.A, G1.A)

    return evall(gma, gmb, ma, mb)


@ex.capture
def eval_klaus(Ae, Be, gma, gmb, maxiter, lalpha):
    A = e_to_G(Ae)
    B = e_to_G(Be)
    L = similarities_preprocess.create_L(A, B, alpha=lalpha)
    S = similarities_preprocess.create_S(A, B, L)
    li, lj, w = sps.find(L)

    ma, mb = klaus.main(S, w, li, lj, a=0, maxiter=maxiter)

    return evall(gma, gmb, ma-1, mb-1)


@ex.capture
def eval_gwl(Ae, Be, gma, gmb):
    n = np.amax(Ae) + 1
    m = np.amax(Be) + 1

    data = {
        'src_index': {float(i): i for i in range(n)},
        'src_interactions': Ae.tolist(),
        'tar_index': {float(i): i for i in range(m)},
        'tar_interactions': Be.tolist(),
        'mutual_interactions': None
    }

    gwl.main(data)

    return (0, 0)


@ex.capture
def eval_isorank(Ae, Be, gma, gmb, maxiter, lalpha):
    A = e_to_G(Ae)
    B = e_to_G(Be)
    L = similarities_preprocess.create_L(A, B, alpha=lalpha)
    S = similarities_preprocess.create_S(A, B, L)
    li, lj, w = sps.find(L)

    ma, mb = isorank.main(S, w, li, lj, a=0.2, b=0.8,
                          alpha=None, rtype=1, maxiter=maxiter)

    return evall(gma, gmb, ma-1, mb-1)


@ex.capture
def eval_grasp(Ae, Be, gma, gmb):
    G1 = e_to_G(Ae)
    G2 = e_to_G(Be)

    ma, mb = grasp.main(G2.A, G1.A, alg=2, base_align=True)
    # ma, mb = grasp.main(G2, G1, alg=2, base_align=True)

    return evall(gma, gmb, ma, mb)


@ex.automain
def main(Ae, Be, gma, gmb):
    np.set_printoptions(threshold=np.inf)
    print(np.array([gma, gmb]).T)

    results = np.array([
        eval_regal(),
        eval_eigenalign(),
        eval_conealign(),
        eval_NSD(),
        eval_grasp(),

        # eval_gwl(),

        # eval_netalign(),
        # eval_klaus(),
        # eval_isorank(),
    ])

    print("\n####################################\n\n")
    print(results)

    # S = "data/karol/S.txt"
    # li = "data/karol/li.txt"
    # lj = "data/karol/lj.txt"

    # S = e_to_G(np.loadtxt(S, int))
    # li = np.loadtxt(li, int)
    # lj = np.loadtxt(lj, int)
    # li -= 1
    # lj -= 1
    # w = np.ones(len(li))
