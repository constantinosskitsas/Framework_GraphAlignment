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


def iter_name(amount, prefix=""):
    return [f"{prefix}{i}" for i in range(1, amount+1)]


@ex.capture
def plotS_G(S_G, _log):
    for gi in S_G:
        for g in gi:
            try:
                _log.debug([len(x)
                            for x in {frozenset(g.nodes[v]["community"]) for v in g}])
            except Exception:
                pass
            plotG(g, 'Src')


@ ex.capture
def plot_G(G):
    for gi in G:
        for ni in gi:
            for g in ni:
                Src, Tar, _ = g
                plotG(Src.tolist(), 'Src', False)
                plotG(Tar.tolist(), 'Tar')


@ ex.capture
def savexls(res5, _run, prefix="", graph_names=None, acc_names=None, alg_names=None, xls_type=1):
    graph_names = iter_name(
        res5.shape[0], "g") if graph_names is None else graph_names
    acc_names = iter_name(
        res5.shape[4], "acc") if acc_names is None else acc_names
    alg_names = iter_name(
        res5.shape[3], "alg") if alg_names is None else alg_names

    output_path = f"runs/{_run._id}/res"
    os.makedirs(output_path, exist_ok=True)

    for graph_number, res4 in enumerate(res5):
        writer = pd.ExcelWriter(
            f"{output_path}/{prefix}_{graph_names[graph_number]}.xlsx", engine='openpyxl')

        for noise_level, res3 in enumerate(res4):
            if xls_type == 1:
                for i in range(res3.shape[2]):
                    sn = str(acc_names[i])
                    rownr = (writer.sheets[sn].max_row +
                             1) if sn in writer.sheets else 0
                    pd.DataFrame(
                        res3[:, :, i],
                        index=[f'it{j+1}' for j in range(res3.shape[0])],
                        columns=[str(alg_names[j])
                                 for j in range(res3.shape[1])],
                    ).to_excel(writer,
                               sheet_name=sn,
                               startrow=rownr,
                               )
            elif xls_type == 2:
                for i in range(res3.shape[1]):
                    sn = alg_names[i]
                    rownr = (writer.sheets[sn].max_row +
                             1) if sn in writer.sheets else 0
                    pd.DataFrame(
                        res3[:, i, :],
                        index=[f'it{j+1}' for j in range(res3.shape[0])],
                        columns=[acc_names[j] for j in range(res3.shape[2])],
                    ).to_excel(writer,
                               sheet_name=sn,
                               startrow=rownr,
                               )

        writer.save()


@ ex.capture
def saveexls(res5, _run, dim1, dim2, dim3, dim4, dim5, prefix=""):

    output_path = f"runs/{_run._id}/res"
    os.makedirs(output_path, exist_ok=True)

    for i1, res4 in enumerate(res5):
        writer = pd.ExcelWriter(
            f"{output_path}/{prefix}_{dim1[i1]}.xlsx", engine='openpyxl')

        for i2, res3 in enumerate(res4):
            for _, res2 in enumerate(res3):
                # for i in range(res3.shape[2]):

                sn = str(dim2[i2])
                rownr = (writer.sheets[sn].max_row +
                         1) if sn in writer.sheets else 0
                pd.DataFrame(
                    res2,
                    index=dim4,
                    columns=dim5,
                ).to_excel(writer,
                           sheet_name=sn,
                           startrow=rownr,
                           )

        writer.save()


@ex.capture
def plotrees(res4, _run, dim1, dim2, dim3, dim4, prefix="", xlabel="Noise level", ylabel="Accuracy"):

    output_path = f"runs/{_run._id}/res"
    os.makedirs(output_path, exist_ok=True)

    for i1, res3 in enumerate(res4):
        for i2, res2 in enumerate(res3):
            plt.figure()
            for i3, res1 in enumerate(res2):
                if np.all(res1 >= 0):
                    plt.plot(dim4, res1, label=dim3[i3])
            plt.xlabel(xlabel)
            plt.xticks(dim4)
            plt.ylabel(ylabel)
            plt.ylim([-0.1, 1.1])
            plt.legend()
            plt.savefig(
                f"{output_path}/res_{dim1[i1]}_{dim2[i2]}.png")

        # elif plot_type == 3:
        #     for n in range(plots.shape[0]):
        #         plt.figure()
        #         for i in range(plots.shape[2]):
        #             vals = plots[n, :, i]
        #             if np.all(vals >= 0):
        #                 plt.plot(alg_names, vals, label=acc_names[i])
        #         plt.xlabel(xlabel)
        #         plt.xticks(alg_names)
        #         plt.ylabel(ylabel)
        #         plt.ylim([-0.1, 1.1])
        #         plt.legend()
        #         plt.savefig(
        #             f"{output_path}/res_{graph_names[graph_number]}_{noises[n]}.png")


@ ex.capture
def plotres(res5, output_path, run, noises, graph_names=None, acc_names=None, alg_names=None, plot_type=1, xlabel="Noise level", ylabel="Accuracy"):
    graph_names = iter_name(
        res5.shape[0], "g") if graph_names is None else graph_names
    acc_names = iter_name(
        res5.shape[4], "acc") if acc_names is None else acc_names
    alg_names = iter_name(
        res5.shape[3], "alg") if alg_names is None else alg_names

    for graph_number, res4 in enumerate(res5):
        plots = np.mean(res4, axis=1)

        if plot_type == 1:
            plt.figure()
            for alg in range(plots.shape[1]):
                vals = plots[:, alg, 0]
                plt.plot(noises, vals, label=alg_names[run[alg]])
            plt.xlabel(xlabel)
            plt.xticks(noises)
            plt.ylabel(ylabel)
            plt.ylim([-0.1, 1.1])
            plt.legend()
            plt.savefig(f"{output_path}/res_{graph_names[graph_number]}.png")

        elif plot_type == 2:
            for alg in range(plots.shape[1]):
                plt.figure()
                for i in range(plots.shape[2]):
                    vals = plots[:, alg, i]
                    if np.all(vals >= 0):
                        plt.plot(noises, vals, label=acc_names[i])
                plt.xlabel(xlabel)
                plt.xticks(noises)
                plt.ylabel(ylabel)
                plt.ylim([-0.1, 1.1])
                plt.legend()
                plt.savefig(
                    f"{output_path}/res_{graph_names[graph_number]}_{alg_names[run[alg]]}.png")

        elif plot_type == 3:
            for n in range(plots.shape[0]):
                plt.figure()
                for i in range(plots.shape[2]):
                    vals = plots[n, :, i]
                    if np.all(vals >= 0):
                        plt.plot(alg_names, vals, label=acc_names[i])
                plt.xlabel(xlabel)
                plt.xticks(alg_names)
                plt.ylabel(ylabel)
                plt.ylim([-0.1, 1.1])
                plt.legend()
                plt.savefig(
                    f"{output_path}/res_{graph_names[graph_number]}_{noises[n]}.png")
