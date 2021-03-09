from algorithms import regal, eigenalign, conealign, netalign, NSD, klaus, gwl, isorank
from data import similarities_preprocess
# from data import ReadFile
# from evaluation import evaluation, evaluation_design
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
    G = sps.csr_matrix((np.ones(e.shape[0]), e.T), shape=(n, n), dtype=int)
    G += G.T
    G.data = G.data.clip(0, 1)
    return G.A


def e_to_Adj(e1, e2):
    adj1 = sps.kron([[1, 0], [0, 0]], e_to_G(e1))
    adj2 = sps.kron([[0, 0], [0, 1]], e_to_G(e2))
    adj = adj1 + adj2
    return adj.A


@ex.config
def global_config():
    noise_level = 1
    edges = 1
    _lim = 0

    data = f"data/noise_level_{noise_level}/edges_{edges}.txt"
    target = "data/arenas_orig.txt"
    gt = f"data/noise_level_{noise_level}/gt_{edges}.txt"

    # gma, gmb = ReadFile.gt1(gt)
    # G1 = ReadFile.edgelist_to_adjmatrix1(data1)
    # G2 = ReadFile.edgelist_to_adjmatrix1(data2)
    # adj = ReadFile.edgelist_to_adjmatrixR(data1, data2)

    gma, gmb = np.loadtxt(gt, int).T
    Ae = np.loadtxt(data, int)
    Be = np.loadtxt(target, int)

    G1 = e_to_G(Ae)
    G2 = e_to_G(Be)
    adj = e_to_Adj(Ae, Be)


@ex.named_config
def demo():
    _lim = 200
    data = "data/arenas_orig.txt"

    Ae = np.loadtxt(data, int)
    # Ae = np.random.permutation(np.amax(Ae)+1)[Ae]

    gma = np.arange(_lim)
    gmb = np.random.permutation(_lim)

    Ae = Ae[np.where(Ae < _lim, True, False).all(axis=1)]
    Be = gmb[Ae]

    G1 = e_to_G(Ae)
    G2 = e_to_G(Be)
    adj = e_to_Adj(Ae, Be)


@ex.capture
def eval_regal(Ae, Be, gma, gmb, adj):
    alignmatrix = regal.main(adj)
    # ma, mb = evaluation.transformRAtoNormalALign(alignmatrix)
    # acc = evaluation.accuracy(gma, gmb, mb, ma)
    # print("acc:", acc)

    ma = np.arange(alignmatrix.shape[0])
    mb = alignmatrix.argmax(1).A1
    return evall(gma, gmb, ma, mb)


@ex.capture
def eval_eigenalign(Ae, Be, gma, gmb, G1, G2):
    ma, mb, _, _ = eigenalign.main(G1, G2, 8, "lowrank_svd_union", 3)
    # acc = evaluation.accuracy(gma+1, gmb+1, mb, ma)
    # print("acc:", acc)

    return evall(gma, gmb, ma-1, mb-1)


@ex.capture
def eval_conealign(Ae, Be, gma, gmb, G1, G2):
    alignmatrix = conealign.main(G1, G2)
    # ma, mb = evaluation.transformRAtoNormalALign(alignmatrix)
    # acc = evaluation.accuracy(gma, gmb, mb, ma)
    # print("acc:", acc)

    ma = np.arange(alignmatrix.shape[0])
    mb = alignmatrix.argmax(1).A1
    return evall(gma, gmb, ma, mb)


@ex.capture
def eval_netalign(Ae, Be, gma, gmb):
    Ai, Aj = Ae.T
    n = max(max(Ai), max(Aj)) + 1
    nedges = len(Ai)
    Aw = np.ones(nedges)
    A = sps.csr_matrix((Aw, (Ai, Aj)), shape=(n, n), dtype=int)
    A = A + A.T

    Bi, Bj = Be.T
    m = max(max(Bi), max(Bj)) + 1
    medges = len(Bi)
    Bw = np.ones(medges)
    B = sps.csr_matrix((Bw, (Bi, Bj)), shape=(m, m), dtype=int)
    B = B + B.T

    L = similarities_preprocess.create_L(A, B, alpha=2)
    S = similarities_preprocess.create_S(A, B, L)
    li, lj, w = sps.find(L)

    ma, mb = netalign.main(S, w, li, lj, a=0)
    return evall(gma, gmb, ma, mb)


@ex.capture
def eval_NSD(Ae, Be, gma, gmb, G1, G2):
    ma, mb = NSD.run(G2, G1)
    # acc = evaluation.accuracy(gma, gmb, mb, ma)
    # print(acc)

    return evall(gma, gmb, ma, mb)


@ex.capture
def eval_klaus(Ae, Be, gma, gmb):
    Ai, Aj = Ae.T
    n = max(max(Ai), max(Aj)) + 1
    nedges = len(Ai)
    Aw = np.ones(nedges)
    A = sps.csr_matrix((Aw, (Ai, Aj)), shape=(n, n), dtype=int)
    A = A + A.T

    Bi, Bj = Be.T
    m = max(max(Bi), max(Bj)) + 1
    medges = len(Bi)
    Bw = np.ones(medges)
    B = sps.csr_matrix((Bw, (Bi, Bj)), shape=(m, m), dtype=int)
    B = B + B.T

    L = similarities_preprocess.create_L(A, B, alpha=4)
    S = similarities_preprocess.create_S(A, B, L)
    li, lj, w = sps.find(L)

    # S = "data/karol/S.txt"
    # li = "data/karol/li.txt"
    # lj = "data/karol/lj.txt"

    # S = e_to_G(np.loadtxt(S, int))
    # li = np.loadtxt(li, int)
    # lj = np.loadtxt(lj, int)
    # li -= 1
    # lj -= 1
    # w = np.ones(len(li))

    ma, mb = klaus.main(S, w, li, lj, a=0, maxiter=10)

    return evall(gma, gmb, ma, mb)


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
def eval_isorank(Ae, Be, gma, gmb):
    Ai, Aj = Ae.T
    n = max(max(Ai), max(Aj)) + 1
    nedges = len(Ai)
    Aw = np.ones(nedges)
    A = sps.csr_matrix((Aw, (Ai, Aj)), shape=(n, n), dtype=int)
    A = A + A.T

    Bi, Bj = Be.T
    m = max(max(Bi), max(Bj)) + 1
    medges = len(Bi)
    Bw = np.ones(medges)
    B = sps.csr_matrix((Bw, (Bi, Bj)), shape=(m, m), dtype=int)
    B = B + B.T

    L = similarities_preprocess.create_L(B, A, alpha=2)
    S = similarities_preprocess.create_S(B, A, L)
    li, lj, w = sps.find(L)

    a = 0.2
    b = 0.8
    ma, mb = isorank.main(S, w, a, b, li, lj, 0)

    # acc = evaluation.accuracy(gma+1, gmb+1, mb, ma)
    # print(acc)

    return evall(gma, gmb, ma-1, mb-1)


@ex.automain
def main(Ae, Be, gma, gmb, G1, G2, adj):
    np.set_printoptions(threshold=np.inf)
    print(np.array([gma, gmb]).T)

    results = np.array([
        eval_regal(),
        eval_eigenalign(),
        eval_conealign(),
        eval_netalign(),
        eval_NSD(),
        # eval_klaus(),
        # eval_gwl(),
        # eval_isorank(),
    ])

    print("\n####################################\n\n")
    print(results)
