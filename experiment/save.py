from . import ex
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import collections
import os


@ex.capture
def plotG(G, name="", end=True, circular=False):
    G = nx.Graph(G)

    plt.figure(name)

    if len(G) <= 200:
        kwargs = {}
        if circular:
            kwargs = dict(pos=nx.circular_layout(G),
                          node_color='r', edge_color='b')
        plt.subplot(211)
        nx.draw(G, **kwargs)

        plt.subplot(212)

    degree_sequence = sorted([d for n, d in G.degree()],
                             reverse=True)  # degree sequence
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())
    plt.bar(deg, cnt, width=0.80, color="b")

    # print(degreeCount)
    plt.title(
        f"{name} Degree Histogram.\nn = {len(G)}, e = {len(G.edges)}, maxd = {deg[0]}, disc = {degreeCount[0]}")
    plt.ylabel("Count")
    plt.xlabel("Degree")
    # fig, ax = plt.subplots()
    # ax.set_xticks([d + 0.4 for d in deg])
    # ax.set_xticklabels(deg)

    plt.show(block=end)


@ex.capture
def plotS_G(S_G, _log, graph_names):
    for i, gi in enumerate(S_G):
        for g in gi:
            try:
                _log.debug([len(x)
                            for x in {frozenset(g.nodes[v]["community"]) for v in g}])
            except Exception:
                pass
            try:
                g_cc = len(max(nx.connected_components(g), key=len))
                _log.debug([g_cc, len(g.nodes)])
            except Exception:
                pass
            try:
                # plotG(g, 'Src')
                plotG(g, graph_names[i])
            except Exception:
                pass


@ ex.capture
def plot_G(G):
    for gi in G:
        for ni in gi:
            for g in ni:
                Src, Tar, _ = g
                plotG(Src.tolist(), 'Src', False)
                plotG(Tar.tolist(), 'Tar')


@ ex.capture
def saveexls(res4, dim1, dim2, dim3, dim4, filename):

    with pd.ExcelWriter(f"{filename}.xlsx") as writer:
        for i1, res3 in enumerate(res4):
            index = pd.MultiIndex.from_product(
                [dim2, dim3], names=["", ""]
            )
            pd.DataFrame(
                res3.reshape(-1, res3.shape[-1]),
                index=index,
                columns=dim4,
            ).to_excel(writer, sheet_name=str(dim1[i1]))


@ex.capture
def plotrees(res3, dim1, dim2, dim3, filename, xlabel="Noise level", plot_type=1):

    for i1, res2 in enumerate(res3):
        plt.figure()
        for i2, res1 in enumerate(res2):
            if np.all(res1 >= 0):
                plt.plot(dim3, res1, label=dim2[i2])
        plt.xlabel(xlabel)
        plt.xticks(dim3)
        if plot_type == 1:
            plt.ylabel("Accuracy")
            plt.ylim([-0.1, 1.1])
        else:
            plt.ylabel("Time[s]")
            # plt.yscale('log')

        plt.legend()
        plt.savefig(
            f"{filename}_{dim1[i1]}.png")


def squeeze(res, dims, sq):

    try:
        res = np.squeeze(res, axis=sq)
        del dims[sq]
    except Exception:
        pass

    return res, dims


def trans(res, dims, T):
    return res.transpose(*T), [dims[i] for i in T]


def save_rec(res, dims, filename, plot_type=1):
    if len(res.shape) > 4:
        for _dim, _res in zip(dims[0], res):
            save_rec(_res, dims[1:], f"{filename}_{_dim}", plot_type)
    else:
        saveexls(res, filename=filename,
                 dim1=dims[0],
                 dim2=dims[1],
                 dim3=dims[2],
                 dim4=dims[3],
                 )

        plotrees(np.mean(res, axis=3), filename=filename, plot_type=plot_type,
                 dim1=dims[0],
                 dim2=dims[1],
                 dim3=dims[2],
                 )


@ ex.capture
def save(time5, res6, output_path, noises, iters, algs, acc_names, graph_names, mt_names=["mt"], mtsq=True, acsq=True, s_trans=None):

    T = [0, 3, 4, 5, 1, 2]

    dims = [
        graph_names,
        noises,
        list(range(1, iters+1)),
        [a[3] for a in algs],
        mt_names,
        acc_names
    ]

    time6 = np.expand_dims(time5, axis=-1)

    # (g,n,i,alg,mt,acc)
    res, dims = trans(res6, dims, T)
    time, _ = trans(time6, list(range(len(T))), T)
    # (g,alg,mt,acc,n,i)

    time_alg = time.take(indices=[0], axis=2)
    ta_dims = dims.copy()
    time_m = time.take(indices=range(1, time.shape[2]), axis=2)
    tm_dims = dims.copy()

    if acsq:
        res, dims = squeeze(res, dims, 3)
        time_alg, ta_dims = squeeze(time_alg, ta_dims, 3)
        time_m, tm_dims = squeeze(time_m, tm_dims, 3)

    if mtsq:
        res, dims = squeeze(res, dims, 2)
        time_alg, ta_dims = squeeze(time_alg, ta_dims, 2)
        time_m, tm_dims = squeeze(time_m, tm_dims, 2)

    if s_trans is not None:
        res, dims = trans(res, dims, s_trans)  # (g,alg,n,i)
        time_alg, ta_dims = trans(time_alg, ta_dims, s_trans)  # (g,alg,n,i)
        time_m, tm_dims = trans(time_m, tm_dims, s_trans)  # (g,alg,n,i)

    print(res.shape, dims)
    print(time_alg.shape, ta_dims)
    print(time_m.shape, tm_dims)

    save_rec(res, dims, f"{output_path}/acc")
    save_rec(time_alg, ta_dims, f"{output_path}/time_alg", plot_type=2)
    save_rec(time_m, tm_dims, f"{output_path}/time_matching", plot_type=2)
