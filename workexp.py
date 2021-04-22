from sacred import Experiment
from algorithms import regal, eigenalign, conealign, netalign, NSD, klaus, gwl, grasp2 as grasp, isorank2 as isorank, bipartitewrapper as bmw
import algorithms

from data import similarities_preprocess
# from scipy.io import loadmat, savemat
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
import datetime
import logging
import time
from sacred.observers import FileStorageObserver


from utils import *

ex = Experiment("ex")

ex.observers.append(FileStorageObserver('runs'))

# create logger
logger = logging.getLogger('e')
logger.setLevel(logging.DEBUG)
logger.propagate = False

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# create formatter
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)

ex.logger = logger


@ex.config
def global_config():

    GW_args = {
        'opt_dict': {
            'epochs': 1,
            'batch_size': 1000000,
            'use_cuda': False,
            'strategy': 'soft',
            # 'strategy': 'hard',
            # 'beta': 0.1,
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
            # 'loss_type': 'MSE',
            'loss_type': 'L2',
            'cost_type': 'cosine',
            # 'cost_type': 'RBF',
            'ot_method': 'proximal'
        },
        # 'lr': 0.001,
        'lr': 1e-3,
        # 'gamma': 0.01,
        # 'gamma': None,
        'gamma': 0.8,
        'max_cpu': 4
    }

    CONE_args = {
        'dim': 128,  # clipped by Src[0] - 1
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
        'alpha': None,  # 0.6 in full
        'tol': 1e-12,
        'maxiter': 100
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
        'rtype': 2,
        'maxiter': 100,
        'verbose': True
    }

    GW_mtype = 2
    CONE_mtype = 3
    GRASP_mtype = -4
    REGAL_mtype = 1
    LREA_mtype = 3
    NSD_mtype = 2
    ISO_mtype = 2
    NET_mtype = 3
    KLAU_mtype = 3

    algs = [
        (gwl, GW_args, GW_mtype),
        (conealign, CONE_args, CONE_mtype),
        (grasp, GRASP_args, GRASP_mtype),
        (regal, REGAL_args, REGAL_mtype),

        (eigenalign, LREA_args, LREA_mtype),
        (NSD, NSD_args, NSD_mtype),
        (isorank, ISO_args, ISO_mtype),

        (netalign, NET_args, NET_mtype),
        (klaus, KLAU_args, KLAU_mtype)
    ]

    # mtype = [
    #     1,      # gwl
    #     2,      # conealign
    #     3,      # grasp
    #     0,      # regal

    #     2,      # eigenalign
    #     1,      # NSD
    #     1,      # isorank

    #     2,      # netalign
    #     2,      # klaus
    # ]

    run = [
        0,      # gwl
        1,      # conealign
        2,      # grasp
        3,      # regal

        4,      # eigenalign
        5,      # NSD
        6,      # isorank

        # 7,      # netalign
        # 8,      # klaus
    ]

    prep = False  # for prep run with full
    lalpha = mind = None

    verbose = False
    mnc = True
    save = False
    plot = False

    iters = 1

    graphs = [
        # (lambda x: nx.Graph([[0, 1]]), ("undefined",))
        (nx.powerlaw_cluster_graph, (50, 5, 0.5))
    ]

    noise_level = _mtype = None
    # no_disc = True
    noises = [
        {'target_noise': noise_level},
        {'target_noise': noise_level, 'refill': True},
        {'source_noise': noise_level, 'target_noise': noise_level},
    ]

    output_path = "results/_" + datetime.datetime.now().strftime("%Y-%m-%d_%H;%M;%S,%f")


@ex.named_config
def gwcost():
    GW_args = {
        'opt_dict': {
            'epochs': 10
        }
    }

    # GW_mtype = -4


@ex.named_config
def mtype():
    _mtype = 0
    GW_mtype = CONE_mtype = GRASP_mtype = REGAL_mtype = LREA_mtype = NSD_mtype = ISO_mtype = NET_mtype = KLAU_mtype = _mtype


@ex.named_config
def full():

    ISO_args = {
        'alpha': 0.6
    }

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


@ex.named_config
def debug():

    verbose = True
    save = True
    plot = True


@ex.named_config
def win():

    GRASP_mtype = -2


@ex.named_config
def fast():

    GW_args = {
        'opt_dict': {
            'epochs': 1,
            'outer_iteration': 40,
            'sgd_iteration': 30,
        },
        'hyperpara_dict': {
            'dimension': 5
        },
        'max_cpu': 0
    }

    GRASP_args = {
        'n_eig': 50,
        'k': 5
    }

    CONE_args = {
        'dim': 16
    }

    run = [
        0,      # gwl
        1,      # conealign,
        2,      # grasp,
        3,      # regal,
    ]

    mnc = False


@ex.capture
def evall(ma, mb, Src, Tar, Gt, output_path, verbose, mnc, save, _log, alg='NoName', eval_type=0):

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

    accs = res[:, 0]
    best = np.argmax(accs)

    if max(accs) < 0:
        if eval_type:
            prefix = "#"
        else:
            _log.warning("misleading evaluation")
            prefix = "!"
    elif eval_type and eval_type != best:
        _log.warning("eval_type mismatch")
        prefix = "%"
    else:
        prefix = ""

    acc, accb, alignment = res[eval_type]

    acc2 = S3(Src.A, Tar.A, ma, mb)
    acc3 = ICorS3GT(Src.A, Tar.A, ma, mb, gmb, True)
    acc4 = ICorS3GT(Src.A, Tar.A, ma, mb, gmb, False)
    if mnc:
        acc5 = score_MNC(Src, Tar, ma, mb)
    else:
        acc5 = -1

    accs = (acc2, acc3, acc4, acc5)

    if save:
        with open(f'{output_path}/{prefix}{alg}_{best}_.txt', 'wb') as f:
            np.savetxt(f, res[:, :2], fmt='%2.3f')
            np.savetxt(f, [accs], fmt='%2.3f')
            np.savetxt(f, [["ma", "mb", "gmab"]], fmt='%5s')
            np.savetxt(f, alignment, fmt='%5d')

    return np.array([acc, *accs])


@ex.capture
def alg_exe(alg, data, args):
    return alg.main(data=data, **args)


@ex.capture
def getmatching(matrix, cost, mt, _log):
    _log.debug("matching type: %s", mt)
    try:
        if mt == 1:
            return colmax(matrix)
        elif mt == 2:
            return superfast(matrix, asc=False)
        elif mt == 3:
            return bmw.getmatchings(matrix)
        elif mt == 4:
            return jv(-np.log(matrix.A))

        elif mt == -1:
            return colmin(cost)
        elif mt == -2:
            return superfast(cost)
        elif mt == -3:
            return bmw.getmatchings(np.exp(-cost.A))
        elif mt == -4:
            return jv(cost)

    except Exception as e:
        _log.exception("")
        return [0], [0]


@ ex.capture
def run_alg(_seed, data, Gt, i, algs, _log, _run):

    random.seed(_seed)
    np.random.seed(_seed)

    alg, args, mt = algs[i]

    _log.debug(f"{' ' + alg.__name__ +' ':#^35}")

    start = time.time()
    res = alg_exe(alg, data, args)
    _run.log_scalar(f"{alg.__name__}.alg", time.time()-start)

    matrix, cost = format_output(res)

    start = time.time()
    ma, mb = getmatching(matrix, cost, mt)
    _run.log_scalar(f"{alg.__name__}.matching", time.time()-start)

    result = evall(ma, mb, data['Src'], data['Tar'], Gt, alg=alg.__name__)

    with np.printoptions(suppress=True, precision=4):
        _log.debug("\n%s", result)
    _log.debug(f"{'#':#^35}")

    return result


@ ex.capture
def preprocess(Src, Tar, lalpha=1, mind=0.00001):
    # L = similarities_preprocess.create_L(Tar, Src, alpha=lalpha, mind=mind)
    L = similarities_preprocess.create_L(Src, Tar, alpha=lalpha, mind=mind)
    # S = similarities_preprocess.create_S(Tar, Src, L)
    S = similarities_preprocess.create_S(Src, Tar, L)
    li, lj, w = sps.find(L)

    return L, S, li, lj, w


@ ex.capture
def run_algs(Src, Tar, Gt, run, prep, plot, _seed, _run, circular=False):

    if plot:
        plotG(Src, 'Src', False, circular=circular)
        plotG(Tar, 'Tar', circular=circular)

    if prep:
        start = time.time()
        L, S, li, lj, w = preprocess(Src, Tar)
        _run.log_scalar("graph.prep", time.time()-start)
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

    # savemat("arenas123.mat", data)

    return np.array([run_alg(_seed, data, Gt, i) for i in run])


@ ex.capture
def init(graphs, noises, iters, no_disc=False):

    G = [
        [
            [
                generate_graphs(alg(*args), no_disc=no_disc, **nargs) for _ in range(iters)
            ] for nargs in noises
        ] for alg, args in graphs
    ]

    randcheck = np.random.rand(1)[0]

    return G, randcheck


@ ex.capture
def run_exp(G, output_path, verbose, _log, _run, _giter=(0, np.inf)):

    first, last = _giter

    _git = 0
    _it = 0
    total_graphs = min(len(G) * len(G[0]), last-first+1)

    for graph_number, g_n in enumerate(G):
        for noise_type, g_it in enumerate(g_n):
            _git += 1
            if _git < first or _git > last:
                continue
            _it += 1

            _log.info("Graph:(%s/%s)", _it, total_graphs)

            writer = pd.ExcelWriter(
                f"{output_path}/res_g{graph_number+1}_n{noise_type+1}.xlsx", engine='openpyxl')

            results = []
            for i, g in enumerate(g_it):
                _log.info("iteration:(%s/%s)", i+1, len(g_it))

                Src_e, Tar_e, Gt = g
                n = Gt[0].size

                src = nx.Graph(Src_e.tolist())
                src_cc = len(max(nx.connected_components(src), key=len))
                src_disc = src_cc < n

                tar = nx.Graph(Tar_e.tolist())
                tar_cc = len(max(nx.connected_components(tar), key=len))
                tar_disc = tar_cc < n

                if (src_disc):
                    _log.warning("Disc. Source: %s < %s", src_cc, n)
                _run.log_scalar("graph.Source.disc", src_disc)
                _run.log_scalar("graph.Source.n", n)
                _run.log_scalar("graph.Source.e", Src_e.shape[0])

                if (tar_disc):
                    _log.warning("Disc. Target: %s < %s", tar_cc, n)
                _run.log_scalar("graph.Target.disc", tar_disc)
                _run.log_scalar("graph.Target.n", n)
                _run.log_scalar("graph.Target.e", Tar_e.shape[0])

                res = run_algs(e_to_G(Src_e, n), e_to_G(Tar_e, n), Gt)

                with np.printoptions(suppress=True, precision=4):
                    _log.info("\n%s", res)

                results.append(res)

            results = np.array(results)

            for i in range(results.shape[2]):
                pd.DataFrame(
                    results[:, :, i],
                    index=[f'it{j+1}' for j in range(results.shape[0])],
                    columns=[f'alg{j+1}' for j in range(results.shape[1])]
                ).to_excel(writer, sheet_name=f"acc{i + 1}")
            writer.save()


@ex.named_config
def playground():

    iters = 1

    graphs = [
        # (nx.newman_watts_strogatz_graph, (100, 3, 0.5)),
        # (nx.watts_strogatz_graph, (100, 10, 0.5)),
        # (nx.gnp_random_graph, (100, 0.9)),
        # (nx.barabasi_albert_graph, (100, 5)),
        (nx.powerlaw_cluster_graph, (213, 2, 0.3)),

        # (nx.relaxed_caveman_graph, (20, 5, 0.2)),

        # (nx.stochastic_block_model, (
        #     [15, 15, 25],
        #     [
        #         [0.25, 0.05, 0.02],
        #         [0.05, 0.35, 0.07],
        #         [0.02, 0.07, 0.40],
        #     ]
        # )),

        # (lambda x: x, ('data/arenas_old/source.txt',)),
        # (lambda x: x, ('data/arenas/source.txt',)),
        # (lambda x: x, ('data/CA-AstroPh/source.txt',)),
        # (lambda x: x, ('data/facebook/source.txt',)),

        # (lambda x: x, ({'dataset': 'arenas_old',
        #                 'edges': 1, 'noise_level': 5},)),

        # (lambda x: x, ({'dataset': 'arenas',
        #                 'edges': 1, 'noise_level': 5},)),

        # (lambda x: x, ({'dataset': 'CA-AstroPh',
        #                 'edges': 1, 'noise_level': 5},)),

        # (lambda x: x, ({'dataset': 'facebook',
        #                 'edges': 1, 'noise_level': 5},)),
    ]

    noise_level = 0.05
    # no_disc = True

    noises = [
        {'target_noise': noise_level},
        # {'target_noise': noise_level, 'refill': True},
        # {'source_noise': noise_level, 'target_noise': noise_level},
    ]

    output_path = "results/pg_" + datetime.datetime.now().strftime("%Y-%m-%d_%H;%M;%S,%f")


@ex.named_config
def exp1s():

    iters = 10

    graphs = [
        (nx.newman_watts_strogatz_graph, (1133, 7, 0.5)),
        (nx.watts_strogatz_graph, (1133, 10, 0.5)),
        (nx.gnp_random_graph, (1133, 0.009)),
        (nx.barabasi_albert_graph, (1133, 5)),
        (nx.powerlaw_cluster_graph, (1133, 5, 0.5)),
    ]

    output_path = "results/exp1s_" + \
        datetime.datetime.now().strftime("%Y-%m-%d_%H;%M;%S,%f")


@ex.named_config
def exp1r():

    iters = 5

    graphs = [
        (lambda x: x, ('data/arenas/source.txt',)),
        # (lambda x: x, ('data/CA-AstroPh/source.txt',)),
        # (lambda x: x, ('data/facebook/source.txt',)),
    ]

    output_path = "results/exp1r_" + \
        datetime.datetime.now().strftime("%Y-%m-%d_%H;%M;%S,%f")


@ex.automain
def main(_config, _run, _log, verbose, output_path, exist_ok=False, nice=10):

    try:
        if not verbose:
            sys.stdout = open(os.devnull, 'w')
            sys.stderr = open(os.devnull, 'w')
            algorithms.GWL.dev.util.logger.disabled = True

        try:
            os.nice(nice)
        except:
            pass

        G, randcheck = init()
        _log.info("randcheck: %s", randcheck)

        os.makedirs(output_path, exist_ok=exist_ok)
        with open(f"{output_path}/config.yaml", "w") as cy:
            conf = {k: v for k, v in _config.items() if
                    not k.endswith("_args") and not k.endswith("_mtype")}

            conf['algs'] = np.array(conf['algs'], dtype=object)[
                conf['run']].tolist()

            conf['_algs'] = conf.pop('algs')
            conf['_graphs'] = conf.pop('graphs')
            conf['_noises'] = conf.pop('noises')
            conf['run_id'] = _run._id

            conf.pop("_giter", None)

            conf['randcheck'] = float(randcheck)

            yaml.dump(conf, cy)

        _log.info("config location: %s", output_path)

        run_exp(G)
    except Exception as e:
        _log.exception("")
