from algorithms import regal, eigenalign, conealign, netalign, NSD, klaus, gwl, grasp, isorank2 as isorank, bipartitewrapper as bmw
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
import networkx as nx

ex = Experiment("experiment")


def plot(cx, filename):
    connects = np.zeros(cx.shape)
    for row, col in zip(cx.row, cx.col):
        connects[row, col] += 1
    plt.imshow(connects)
    plt.savefig(f'results/{filename}.png')
    plt.close('all')


def plotG(G):
    G = nx.Graph(G)
    plt.subplot(121)
    nx.draw(G)   # default spring_layout
    plt.subplot(122)
    nx.draw(G, pos=nx.circular_layout(G),
            node_color='r', edge_color='b')

    plt.show()


def plotGs(left, right, circular=False):
    G_left = nx.Graph(left)
    G_right = nx.Graph(right)
    if circular:
        plt.subplot(121)
        nx.draw(G_left, pos=nx.circular_layout(G_left),
                node_color='r', edge_color='b')
        plt.subplot(122)
        nx.draw(G_right, pos=nx.circular_layout(G_right),
                node_color='r', edge_color='b')
    else:
        plt.subplot(121)
        nx.draw(G_left)
        plt.subplot(122)
        nx.draw(G_right)

    plt.show()


def colmax(matrix):
    ma = np.arange(matrix.shape[0])
    mb = matrix.argmax(1).A1
    return ma, mb


def colmin(matrix):
    ma = np.arange(matrix.shape[0])
    mb = matrix.argmin(1).A1
    return ma, mb


def fast3(l2):
    l2 = l2.A
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
    l2 = l2.A
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


def getmatching(matrix, cost, mtype):
    try:
        # print(cost)
        # print(np.amax(cost))
        # print(cost * -1)
        # print(cost.A * -1 + np.amax(cost))
        if mtype == 0:
            return colmax(matrix)
        elif mtype == 1:
            return fast3(matrix)
        elif mtype == 2:
            return bmw.getmatchings(matrix)
        elif mtype == 3:
            return colmin(cost)
        elif mtype == 4:
            return fast4(cost)
        elif mtype == 5:
            return bmw.getmatchings(sps.csr_matrix(cost.A * -1 + np.amax(cost.A)))
    except Exception as e:
        print(e)
        return [0], [0]


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
                except:
                    pass
                # one_hop_neighbor_counter.append(counterb[count])

            num_stable_neighbor = np.intersect1d(
                new_one_hop_neighbor, np.array(one_hop_neighbor_counter)).shape[0]
            union_align = np.union1d(new_one_hop_neighbor, np.array(
                one_hop_neighbor_counter)).shape[0]

            sim = float(num_stable_neighbor) / union_align
            mnc += sim

        return mnc / countera.size
    except Exception as e:
        return -1


def evall(ma, mb, Tar, Src, Gt, eval_type=0, alg=np.random.rand(), verbose=True):
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
        score_MNC(Tar, Src, ma, mb),
        score_MNC(Tar, Src, mb, ma),
        score_MNC(Src, Tar, ma, mb),
        score_MNC(Src, Tar, mb, ma)
    ])

    acc6 = mnc[eval_type]

    print(f"{' ' + alg +' ':#^35}")
    # print(alignment, end="\n\n")
    print(res[:, :2].astype(float))
    print(mnc.astype(float))
    print(f"{'#':#^35}")

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

    _mtype = None

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

    GRASP_args = {
        'laa': 2,
        'icp': False,
        'icp_its': 3,
        'q': 100,
        'k': 20,
        'n_eig': None,  # Src.shape[0],
        'lower_t': 1.0,
        'upper_t': 50.0,
        'linsteps': True,
        'base_align': True
    }

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

    LREA_args = {
        'iters': 8,
        'method': "lowrank_svd_union",
        'bmatch': 3,
        'default_params': True
    }

    NSD_args = {
        'alpha': 0.5,
        'iters': 10
    }

    ISO_args = {
        # 'alpha': 0.5,
        'alpha': None,
        'tol': 1e-12,
        'maxiter': 1,
        'verbose': True
    }

    NET_args = {
        'a': 1,
        'b': 1,
        'gamma': 0.99,
        'dtype': 2,
        'maxiter': 100,
        'verbose': True
    }

    KLAU_args = {
        'a': 1,
        'b': 1,
        'gamma': 0.4,
        'stepm': 25,
        'rtype': 1,
        'maxiter': 100,
        'verbose': True
    }

    algs = [
        (gwl, GW_args),
        (conealign, CONE_args),
        (grasp, GRASP_args),
        (regal, REGAL_args),

        (eigenalign, LREA_args),
        (NSD, NSD_args),
        (isorank, ISO_args),

        (netalign, NET_args),
        (klaus, KLAU_args)
    ]

    mtype = [
        1,      # gwl
        2,      # conealign,
        3,      # grasp,
        0,      # regal,

        2,      # eigenalign,
        1,      # NSD,
        1,      # isorank,

        2,      # netalign,
        2,      # klaus,
    ]

    if _mtype is not None:
        mtype = [_mtype] * 9

    run = [
        0,      # gwl
        1,      # conealign,
        2,      # grasp,
        3,      # regal,

        4,      # eigenalign,
        5,      # NSD,
        6,      # isorank,

        # 7,      # netalign,
        # 8,      # klaus,
    ]

    prep = False
    lalpha = mind = None
    verbose = True


@ex.named_config
def prep():
    prep = True
    lalpha = 1

    mind = None
    run = [
        0,      # gwl
        1,      # conealign,
        2,      # grasp,
        3,      # regal,

        4,      # eigenalign,
        5,      # NSD,
        6,      # isorank,

        7,      # netalign,
        8,      # klaus,
    ]


def generate_graphs(G, noise=0.05):

    if isinstance(G, dict):
        dataset = G['dataset']
        edges = G['edges']
        noise_level = int(100*noise)

        target = f"data/{dataset}/target.txt"
        source = f"data/{dataset}/noise_level_{noise_level}/edges_{edges}.txt"
        grand_truth = f"data/{dataset}/noise_level_{noise_level}/gt_{edges}.txt"

        Tar_e = np.loadtxt(target, int)
        Src_e = np.loadtxt(source, int)
        gt_e = np.loadtxt(grand_truth, int).T

        Tar = e_to_G(Tar_e)
        Src = e_to_G(Src_e)

        Gt = (
            gt_e[:, gt_e[1].argsort()][0],  # source -> target
            gt_e[:, gt_e[0].argsort()][1]   # target -> source
        )

        return Tar, Src, Gt
    elif isinstance(G, str):
        Tar_e = np.loadtxt(G, int)
    elif isinstance(G, nx.Graph):
        Tar_e = np.array(G.edges)
    else:
        return sps.csr_matrix([]), sps.csr_matrix([]), (np.empty(1), np.empty(1))

    n = np.amax(Tar_e) + 1

    gt_e = np.array((
        np.arange(n),
        np.random.permutation(n)
    ))

    Gt = (
        gt_e[:, gt_e[1].argsort()][0],  # source -> target
        gt_e[:, gt_e[0].argsort()][1]   # target -> source
    )

    r = np.random.sample(Tar_e.shape[0])
    Src_e = Tar_e[r > noise]
    Src_e = Gt[0][Src_e]

    return e_to_G(Tar_e), e_to_G(Src_e), Gt


@ex.named_config
def demo2():
    Tar, Src, Gt = dem()
    # plt.subplot(121)
    # nx.draw(G)   # default spring_layout
    # plt.subplot(122)
    # nx.draw(G, pos=nx.circular_layout(G),
    #         node_color='r', edge_color='b')

    # plt.show()


def format_output(res):

    if isinstance(res, tuple):
        matrix, cost = res
    else:
        matrix = res
        cost = None

    try:
        matrix = sps.csr_matrix(matrix)
    except Exception as e:
        matrix = None
    try:
        cost = sps.csr_matrix(cost)
    except Exception as e:
        cost = None

    # print(matrix)
    # print(type(matrix))
    # try:
    #     print(matrix.shape)
    # except:
    #     pass
    # print(cost)
    # print(type(cost))
    # try:
    #     print(cost.shape)
    # except:
    #     pass

    return matrix, cost


@ex.automain
def main(_config, algs, run, mtype, prep, lalpha, mind, verbose):
    print()

    if not verbose:
        sys.stdout = open(os.devnull, 'w')

    # G = nx.newman_watts_strogatz_graph(1000, 4, 0.5)
    G = nx.watts_strogatz_graph(100, 4, 0.5)
    # G = nx.gnp_random_graph(10, 0.5)  # fast_gnp_random_graph for sparse
    # G = nx.barabasi_albert_graph(100, 10)
    # G = nx.powerlaw_cluster_graph(100, 10, 0.1)

    # G = 'data/arenas_orig/target.txt'

    # G = {'dataset': 'arenas_orig', 'edges': 3}

    Tar, Src, Gt = generate_graphs(G, noise=0.05)

    # plotG(Tar)
    plotGs(Tar, Src, circular=True)

    if prep == True:
        L, S, li, lj, w = preprocess(Src, Tar, lalpha, mind)
    else:
        L = S = li = lj = w = np.empty(1)

    data = {
        'Tar': Tar,
        'Src': Src,
        'L': L,
        'S': S,
        'li': li,
        'lj': lj,
        'w': w
    }

    results = []

    for i in run:
        alg, args = algs[i]
        mt = mtype[i]
        res = alg.main(data=data, **args)
        matrix, cost = format_output(res)
        ma, mb = getmatching(matrix, cost, mt)
        results.append(
            evall(ma, mb, Tar, Src, Gt,
                  alg=alg.__name__, verbose=verbose)
        )

    df = pd.DataFrame(results)
    df.to_csv(f'results/res.csv', index=False)

    sys.stdout = sys.__stdout__
    print("\n")
    print(_config)
    print("\n\n")
    with np.printoptions(precision=4, suppress=True) as a:
        print(np.array(results))


# @ex.capture
# def eval_regal(Tar, Src):


#     matrix = regal.main(Tar, Src, REGAL_args)

#     ma, mb = colmax(matrix)
#     # ma, mb = bmw.getmatchings(matrix)
#     # ma, mb = fast3(matrix.A)

#     return evall(ma, mb, matrix,
#                  alg=inspect.currentframe().f_code.co_name)


# @ex.capture
# def eval_eigenalign(Tar, Src):


#     matrix = eigenalign.main(Tar.A, Src.A, **LREA_args)

#     # ma, mb = colmax(matrix)
#     # ma, mb = fast3(matrix.A)
#     ma, mb = bmw.getmatchings(matrix)

#     return evall(ma, mb, matrix,
#                  alg=inspect.currentframe().f_code.co_name)


# @ex.capture
# def eval_conealign(Tar, Src, Gt):


#     matrix = conealign.main(Tar.A, Src.A, CONE_args)

#     ma, mb = colmax(matrix)
#     # ma, mb = fast3(matrix.A)
#     # ma, mb = bmw.getmatchings(matrix)

#     return evall(ma, mb, matrix, Tar, Src, Gt,
#                  alg=inspect.currentframe().f_code.co_name)


# @ex.capture
# def eval_NSD(Tar, Src):


#     matrix = NSD.run(Tar.A, Src.A, **NSD_args)

#     # ma, mb = colmax(sps.csr_matrix(matrix))
#     ma, mb = fast3(matrix)
#     # ma, mb = bmw.getmatchings(matrix)

#     return evall(ma, mb, matrix,
#                  alg=inspect.currentframe().f_code.co_name)


# @ex.capture
# def eval_grasp(Tar, Src):


#     cost = grasp.main(Tar.A, Src.A, GRASP_args)
#     # cost = grasp.main(Tar, Src, GRASP_args)
#     # cost = grasp.main(Src.A, Tar.A, GRASP_args)
#     # cost = grasp.main(Src, Tar, GRASP_args)

#     # ma, mb = colmin(sps.csr_matrix(cost))
#     ma, mb = fast4(cost)
#     # ma, mb = bmw.getmatchings(cost)

#     return evall(ma, mb, cost,
#                  alg=inspect.currentframe().f_code.co_name)


# @ex.capture
# def eval_gwl(Tar, Src):


#     matrix, cost = gwl.main(Tar, Src, **GW_args)

#     # ma, mb = colmin(sps.csr_matrix(cost))
#     # ma, mb = fast4(cost)

#     # ma, mb = colmax(sps.csr_matrix(matrix))
#     ma, mb = fast3(matrix)
#     # ma, mb = bmw.getmatchings(matrix)

#     return evall(ma, mb, matrix,
#                  alg=inspect.currentframe().f_code.co_name)


# @ex.capture
# def eval_isorank(Tar, Src, L, S, w, li, lj):


#     matrix = isorank2.main(Tar.A, Src.A, **ISO_args)

#     # ma, mb = colmax(sps.csr_matrix(matrix))
#     ma, mb = fast3(matrix)
#     # ma, mb = bmw.getmatchings(matrix)

#     return evall(ma, mb, matrix,
#                  alg=inspect.currentframe().f_code.co_name)


# @ex.capture
# def eval_netalign(S, w, li, lj, maxiter):


#     matrix = netalign.main(S, w, li, lj, **NET_args)

#     # ma, mb = colmax(matrix)
#     # ma, mb = fast3(matrix.A)
#     ma, mb = bmw.getmatchings(matrix)

#     return evall(ma, mb, matrix,
#                  alg=inspect.currentframe().f_code.co_name)


# @ex.capture
# def eval_klaus(S, w, li, lj, maxiter):


#     matrix = klaus.main(S, w, li, lj, **KLAU_args)

#     # ma, mb = colmax(matrix)
#     # ma, mb = fast3(matrix.A)
#     ma, mb = bmw.getmatchings(matrix)

#     return evall(ma, mb, matrix,
#                  alg=inspect.currentframe().f_code.co_name)
