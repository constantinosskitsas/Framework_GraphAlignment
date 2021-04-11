from sacred import Experiment
from algorithms import regal, eigenalign, conealign, netalign, NSD, klaus, gwl, grasp, isorank2 as isorank, bipartitewrapper as bmw

# from data import similarities_preprocess
# from scipy.io import loadmat
# import inspect
# import matplotlib.pyplot as plt
# from data import ReadFile
import pandas as pd
# from math import log2

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import collections
import scipy.sparse as sps
import sys
import os
import random
import yaml

from utils import *

ex = Experiment("experiment")


@ex.config
def global_config():

    _mtype = None

    GW_args = {
        'opt_dict': {
            'epochs': 5,
            'batch_size': 1000000,
            'use_cuda': False,
            'strategy': 'soft',
            # 'strategy': 'hard',
            #'beta': 0.1,
            'beta': 1e-1,
            'outer_iteration': 400,  # M
            'inner_iteration': 1,  # N
            'sgd_iteration': 300,
            'prior': False,
            'prefix': 'results',
            'display': False
        },
        'hyperpara_dict': {
            'dimension': 90,
            #'loss_type': 'MSE',
            'loss_type': 'L2',
            'cost_type': 'cosine',
            #'cost_type': 'RBF',
            'ot_method': 'proximal'
        },
        #'lr': 0.001,
        'lr': 1e-3,
        #'gamma': 0.01,
        #'gamma': None,
        'gamma': 0.8,
    }

    CONE_args = {
        'dim': 128,
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
        'n_eig': None,  # Src.shape[0] - 1
        'lower_t': 1.0,
        'upper_t': 50.0,
        'linsteps': True,
        'base_align': True
    }

    REGAL_args = {
        'attributes': None,
        'attrvals': 2,
        'dimensions': 128,  # useless
        'k': 10,            # d = klogn
        'untillayer': 2,    # k
        'alpha': 0.01,      # delta
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
        'alpha': 0.8,
        'iters': 10
    }

    ISO_args = {
        'alpha': 0.6,
        # 'alpha': None,
        'tol': 1e-12,
        'maxiter': 100,
        'verbose': True
    }

    NET_args = {
        'a': 1,
        'b': 2,
        'gamma': 0.95,
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
        4,      # gwl
        2,      # conealign
        3,      # grasp
        0,      # regal

        2,      # eigenalign
        1,      # NSD
        1,      # isorank

        2,      # netalign
        2,      # klaus
    ]

    if _mtype is not None:
        mtype = [_mtype] * 9

    run = [
        0,      # gwl
        #1,      # conealign
        #2,      # grasp
        #3,      # regal

        #4,      # eigenalign
        #5,      # NSD
        #6,      # isorank

        # 7,      # netalign
        # 8,      # klaus
    ]

    fast = False

    prep = False
    lalpha = mind = None
    verbose = True


@ex.named_config
def full():
    prep = True
    lalpha = 1

    mind = None
    run = [
        0,      # gwl
        1,      # conealign,
        #2,      # grasp,
        3,      # regal,

        4,      # eigenalign,
        5,      # NSD,
        6,      # isorank,

        7,      # netalign,
        8,      # klaus,
    ]


@ex.named_config
def fast():

    GRASP_args = {
        'laa': 2,
        'icp': False,
        'icp_its': 3,
        'q': 100,
        'k': 20,
        'n_eig': 100,
        'lower_t': 1.0,
        'upper_t': 50.0,
        'linsteps': True,
        'base_align': True
    }

    run = [
        # 0,      # gwl
        # 1,      # conealign,
        2,      # grasp,
        3,      # regal,

        # 4,      # eigenalign,
        5,      # NSD,
        6,      # isorank,
    ]

    fast = True


@ex.capture
def evall(ma, mb, Src, Tar, Gt, eval_type=0, alg='NoName', verbose=True, fast=False):
    sys.stdout = sys.__stdout__
    np.set_printoptions(threshold=100, precision=4, suppress=True)
    # np.set_printoptions(threshold=np.inf)

    print(f"{' ' + alg +' ':#^35}")

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
    print(res[:, :2].astype(float))

    acc3 = S3(Src.A, Tar.A, ma, mb)
    print(acc3)
    acc4 = ICorS3GT(Src.A, Tar.A, ma, mb, gmb, True)
    print(acc4)
    acc5 = ICorS3GT(Src.A, Tar.A, ma, mb, gmb, False)
    print(acc5)
    if fast:
        acc6 = -1
    else:
        acc6 = score_MNC(Src, Tar, ma, mb)
    print(acc6)

    accs = (acc3, acc4, acc5, acc6)

    with open(f'results/{prefix}{alg}_{best}_.txt', 'wb') as f:
        np.savetxt(f, res[:, :2], fmt='%2.3f')
        np.savetxt(f, [accs], fmt='%2.3f')
        if not fast:
            np.savetxt(f, [["ma", "mb", "gmab"]], fmt='%5s')
            np.savetxt(f, alignment, fmt='%5d')

    print(f"{'#':#^35}")

    if not verbose:
        sys.stdout = open(os.devnull, 'w')

    return acc, acc2, *accs


@ex.capture
def run_alg(_seed, data, Gt, i, algs, mtype, verbose):

    random.seed(_seed)
    np.random.seed(_seed)

    alg, args = algs[i]
    mt = mtype[i]
    res = alg.main(data=data, **args)
    matrix, cost = format_output(res)
    ma, mb = getmatching(matrix, cost, mt)
    return evall(ma, mb, data['Src'], data['Tar'], Gt,
                 alg=alg.__name__, verbose=verbose)


@ex.capture
def run_algs(Src, Tar, Gt, algs, run, mtype, prep, lalpha, mind, verbose, _seed, filename="res"):

    if verbose:
        plotG(Src, 'Src', False)
        plotG(Tar, 'Tar')
        # plotGs(Tar, Src, circular=True)

    if prep == True:
        L, S, li, lj, w = preprocess(Src, Tar, lalpha, mind)
    else:
        L = S = sps.eye(1)
        li = lj = w = np.empty(1)

    data = {
        'Src': Src,
        'Tar': Tar,
        'L': L,
        'S': S,
        'li': li,
        'lj': lj,
        'w': w
    }

    results = [run_alg(_seed, data, Gt, i) for i in run]

    df = pd.DataFrame(results)
    df.to_csv(f'results/{filename}.csv', index=False)

    return results


def exp1():
    n = 107
    graphs = [
        (nx.newman_watts_strogatz_graph, (n, 4, 0.5)),
        (nx.watts_strogatz_graph, (n, 10, 0.5)),
        (nx.gnp_random_graph, (n, 0.5)),
        (nx.barabasi_albert_graph, (n, 30)),
        (nx.powerlaw_cluster_graph, (n, 10, 0.5)),
    ]

    noise_level = 0.05

    noises = [
        {'target_noise': noise_level},
        {'target_noise': noise_level, 'refill': True},
        {'source_noise': noise_level, 'target_noise': noise_level},
        # {'source_noise': noise, 'target_noise': noise, 'refill': True},
    ]

    G = [
        [generate_graphs(alg(*args), **nargs) for nargs in noises] for alg, args in graphs
    ]

    for i, gsn in enumerate(G):
        for j, g in enumerate(gsn):
            run_algs(*g, filename=f'res_{i}_{j}', verbose=False)


@ex.capture
def playground():
    G = nx.newman_watts_strogatz_graph(1133, 7, 0.5)
    #G = nx.watts_strogatz_graph(1133, 10, 0.5)
    # G = nx.gnp_random_graph(1133, 0.009)  # fast_gnp_random_graph for sparse
    #G = nx.barabasi_albert_graph(1133, 5)
    # G = nx.powerlaw_cluster_graph(1133, 5, 0.5)

    # G = 'data/arenas_old/source.txt'
    # G = 'data/arenas/source.txt'
    # G = 'data/CA-AstroPh/source.txt'
    # G = 'data/facebook/source.txt'

    #G = {'dataset': 'arenas_old', 'edges': 1, 'noise_level': 5}
    # G = {'dataset': 'arenas', 'edges': 1, 'noise_level': 5}
    # G = {'dataset': 'CA-AstroPh', 'edges': 1, 'noise_level': 5}
    # G = {'dataset': 'facebook', 'edges': 1, 'noise_level': 5}

    noise = 0.1
    Src, Tar, Gt = generate_graphs(
        G,
        # source_noise=noise,
        target_noise=noise,
        # refill=True
    )

    results = run_algs(Src, Tar, Gt)

    sys.stdout = sys.__stdout__
    with np.printoptions(precision=4, suppress=True) as a:
        print(np.array(results))


@ex.automain
def main(_config, verbose):
    print()

    if not verbose:
        sys.stdout = open(os.devnull, 'w')

    playground()
    # exp1()

    # Src, Tar, Gt = init()

    # if verbose:
    #     plotG(Src, 'Src', False)
    #     plotG(Tar, 'Tar')
    #     # plotGs(Tar, Src, circular=True)

    # if prep == True:
    #     L, S, li, lj, w = preprocess(Src, Tar, lalpha, mind)
    # else:
    #     L = S = li = lj = w = np.empty(1)

    # data = {
    #     'Src': Src,
    #     'Tar': Tar,
    #     'L': L,
    #     'S': S,
    #     'li': li,
    #     'lj': lj,
    #     'w': w
    # }

    # results = []

    # for i in run:
    #     # alg, args = algs[i]
    #     # mt = mtype[i]
    #     # res = alg.main(data=data, **args)
    #     # matrix, cost = format_output(res)
    #     # ma, mb = getmatching(matrix, cost, mt)
    #     # results.append(
    #     #     evall(ma, mb, data['Src'], data['Tar'], Gt,
    #     #           alg=alg.__name__, verbose=verbose)
    #     # )
    #     results.append(run_alg(_seed, data, Gt, i))

    # df = pd.DataFrame(results)
    # df.to_csv(f'results/res.csv', index=False)

    with open("config.yaml", "w") as cy:
        yaml.dump(_config, cy)
