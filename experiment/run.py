from . import ex, similarities_preprocess
from evaluation import matching, evaluation
import numpy as np
import networkx as nx
import scipy.sparse as sps
import os
import copy
import time
import gc


def format_output(res):

    if isinstance(res, tuple):
        sim, cost = res
    else:
        sim = res
        cost = None

    # if sim is not None:
    #     sim = sps.csr_matrix(sim)

    # if cost is not None:
    #     cost = sps.csr_matrix(cost)

    if sps.issparse(sim):
        sim = sim.A

    if sps.issparse(cost):
        cost = cost.A

    return sim, cost


# @profile
@ex.capture
def alg_exe(alg, data, args):
    return alg.main(data=data, **args)


@ ex.capture
def run_alg(_alg, _data, Gt, accs, _log, _run, mall):

    # random.seed(_seed)
    # np.random.seed(_seed)

    alg, args, mts, algname = _alg

    _log.debug(f"{f' {algname} ':#^35}")

    data = copy.deepcopy(_data)

    time1 = []

    # gc.disable()
    start = time.time()
    res = alg_exe(alg, data, args)
    time1.append(time.time()-start)
    # gc.enable()
    # gc.collect()

    sim, cost = format_output(res)

    try:
        _run.log_scalar(f"{algname}.sim.size", sim.size)
        _run.log_scalar(f"{algname}.sim.max", sim.max())
        _run.log_scalar(f"{algname}.sim.min", sim.min())
        _run.log_scalar(f"{algname}.sim.avg", sim.data.mean())
    except Exception:
        pass
    try:
        _run.log_scalar(f"{algname}.cost.size", cost.size)
        _run.log_scalar(f"{algname}.cost.max", cost.max())
        _run.log_scalar(f"{algname}.cost.min", cost.min())
        _run.log_scalar(f"{algname}.cost.avg", cost.data.mean())
    except Exception:
        pass

    res2 = []
    for mt in mts:
        alg = f"{algname}_{mt}"
        try:

            start = time.time()
            ma, mb = matching.getmatching(sim, cost, mt)
            elapsed = time.time()-start

            res1 = evaluation.evall(ma, mb, _data['Src'],
                                    _data['Tar'], Gt, alg=alg)
        except Exception:
            if not mall:
                _log.exception("")
            elapsed = -1
            res1 = -np.ones(len(accs))

        time1.append(elapsed)
        res2.append(res1)

    time1 = np.array(time1)
    res2 = np.array(res2)

    with np.printoptions(suppress=True, precision=4):
        _log.debug("\n%s", res2.astype(float))

    _log.debug(f"{'#':#^35}")

    return time1, res2


# @profile
@ ex.capture
def preprocess(Src, Tar, _run):
    start = time.time()
    # L = similarities_preprocess.create_L(Tar, Src)
    L = similarities_preprocess.create_L(Src, Tar)
    # L, _ = regal.main({"Src": Src, "Tar": Tar}, **REGAL_args)
    # L, _ = conealign.main({"Src": Src, "Tar": Tar}, **CONE_args)
    _run.log_scalar("graph.prep.L", time.time()-start)

    start = time.time()
    # S = similarities_preprocess.create_S(Tar, Src, L)
    S = similarities_preprocess.create_S(Src, Tar, L)
    _run.log_scalar("graph.prep.S", time.time()-start)

    li, lj, w = sps.find(L)

    return L, S, li, lj, w


@ ex.capture
def run_algs(g, algs, _log, _run, prep=False, circular=False):

    Src_e, Tar_e, Gt = g
    n = Gt[0].size

    # prefix = f"{output_path}/graphs/{graph_number+1:0>2d}_{noise_level+1:0>2d}_{i+1:0>2d}"
    # Gt_m = np.c_[np.arange(n), Gt[0]]
    # np.savetxt(f"{prefix}_Src.txt", Src_e, fmt='%d')
    # np.savetxt(f"{prefix}_Tar.txt", Tar_e, fmt='%d')
    # np.savetxt(f"{prefix}_Gt.txt", Gt_m, fmt='%d')

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

    Src = e_to_G(Src_e, n)
    Tar = e_to_G(Tar_e, n)

    if prep:
        L, S, li, lj, w = preprocess(Src, Tar)
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

    time2 = []
    res3 = []

    for alg in algs:
        time1, res2 = run_alg(alg, data, Gt)
        time2.append(time1)
        res3.append(res2)

    return np.array(time2), np.array(res3)


def e_to_G(e, n):
    # n = np.amax(e) + 1
    nedges = e.shape[0]
    # G = sps.csr_matrix((np.ones(nedges), e.T), shape=(n, n), dtype=int)
    G = sps.csr_matrix((np.ones(nedges), e.T), shape=(n, n), dtype=np.int8)
    G += G.T
    G.data = G.data.clip(0, 1)
    # return G
    return G.A


@ ex.capture
def run_exp(G, output_path, _log):
    time2 = time3 = time4 = None
    time5 = []
    res3 = res4 = res5 = None
    res6 = []
    try:
        # os.mkdir(f'{output_path}/graphs')

        for graph_number, g_n in enumerate(G):

            _log.info("Graph:(%s/%s)", graph_number + 1, len(G))

            time4 = []
            res5 = []
            for noise_level, g_it in enumerate(g_n):

                _log.info("Noise_level:(%s/%s)", noise_level + 1, len(g_n))

                time3 = []
                res4 = []
                for i, g in enumerate(g_it):
                    _log.info("iteration:(%s/%s)", i+1, len(g_it))

                    time2, res3 = run_algs(g)

                    with np.printoptions(suppress=True, precision=4):
                        _log.info("\n%s", res3.astype(float))

                    time3.append(time2)
                    res4.append(res3)

                time3 = np.array(time3)
                res4 = np.array(res4)
                with np.printoptions(suppress=True, precision=4):
                    _log.debug("\n%s", res4.astype(float))
                time4.append(time3)
                res5.append(res4)

            time4 = np.array(time4)
            res5 = np.array(res5)
            time5.append(time4)
            res6.append(res5)
    except:
        np.save(f"{output_path}/_time2", np.array(time2))
        np.save(f"{output_path}/_time3", np.array(time3))
        np.save(f"{output_path}/_time4", np.array(time4))
        np.save(f"{output_path}/_res3", np.array(res3))
        np.save(f"{output_path}/_res4", np.array(res4))
        np.save(f"{output_path}/_res5", np.array(res5))
        _log.exception("")

    return np.array(time5), np.array(res6)  # (g,n,i,alg,mt,acc)
