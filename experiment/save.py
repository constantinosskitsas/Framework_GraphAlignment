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
            try:
                plotG(g, 'Src')
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


# @ ex.capture
# def savexls(res5, _run, prefix="", graph_names=None, acc_names=None, alg_names=None, xls_type=1):
#     graph_names = iter_name(
#         res5.shape[0], "g") if graph_names is None else graph_names
#     acc_names = iter_name(
#         res5.shape[4], "acc") if acc_names is None else acc_names
#     alg_names = iter_name(
#         res5.shape[3], "alg") if alg_names is None else alg_names

#     output_path = f"runs/{_run._id}/res"
#     os.makedirs(output_path, exist_ok=True)

#     for graph_number, res4 in enumerate(res5):
#         writer = pd.ExcelWriter(
#             f"{output_path}/{prefix}_{graph_names[graph_number]}.xlsx", engine='openpyxl')

#         for noise_level, res3 in enumerate(res4):
#             if xls_type == 1:
#                 for i in range(res3.shape[2]):
#                     sn = str(acc_names[i])
#                     rownr = (writer.sheets[sn].max_row +
#                              1) if sn in writer.sheets else 0
#                     pd.DataFrame(
#                         res3[:, :, i],
#                         index=[f'it{j+1}' for j in range(res3.shape[0])],
#                         columns=[str(alg_names[j])
#                                  for j in range(res3.shape[1])],
#                     ).to_excel(writer,
#                                sheet_name=sn,
#                                startrow=rownr,
#                                )
#             elif xls_type == 2:
#                 for i in range(res3.shape[1]):
#                     sn = alg_names[i]
#                     rownr = (writer.sheets[sn].max_row +
#                              1) if sn in writer.sheets else 0
#                     pd.DataFrame(
#                         res3[:, i, :],
#                         index=[f'it{j+1}' for j in range(res3.shape[0])],
#                         columns=[acc_names[j] for j in range(res3.shape[2])],
#                     ).to_excel(writer,
#                                sheet_name=sn,
#                                startrow=rownr,
#                                )

#         writer.save()


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


# @ ex.capture
# def plotres(res5, output_path, run, noises, graph_names=None, acc_names=None, alg_names=None, plot_type=1, xlabel="Noise level", ylabel="Accuracy"):
#     graph_names = iter_name(
#         res5.shape[0], "g") if graph_names is None else graph_names
#     acc_names = iter_name(
#         res5.shape[4], "acc") if acc_names is None else acc_names
#     alg_names = iter_name(
#         res5.shape[3], "alg") if alg_names is None else alg_names

#     for graph_number, res4 in enumerate(res5):
#         plots = np.mean(res4, axis=1)

#         if plot_type == 1:
#             plt.figure()
#             for alg in range(plots.shape[1]):
#                 vals = plots[:, alg, 0]
#                 plt.plot(noises, vals, label=alg_names[run[alg]])
#             plt.xlabel(xlabel)
#             plt.xticks(noises)
#             plt.ylabel(ylabel)
#             plt.ylim([-0.1, 1.1])
#             plt.legend()
#             plt.savefig(f"{output_path}/res_{graph_names[graph_number]}.png")

#         elif plot_type == 2:
#             for alg in range(plots.shape[1]):
#                 plt.figure()
#                 for i in range(plots.shape[2]):
#                     vals = plots[:, alg, i]
#                     if np.all(vals >= 0):
#                         plt.plot(noises, vals, label=acc_names[i])
#                 plt.xlabel(xlabel)
#                 plt.xticks(noises)
#                 plt.ylabel(ylabel)
#                 plt.ylim([-0.1, 1.1])
#                 plt.legend()
#                 plt.savefig(
#                     f"{output_path}/res_{graph_names[graph_number]}_{alg_names[run[alg]]}.png")

#         elif plot_type == 3:
#             for n in range(plots.shape[0]):
#                 plt.figure()
#                 for i in range(plots.shape[2]):
#                     vals = plots[n, :, i]
#                     if np.all(vals >= 0):
#                         plt.plot(alg_names, vals, label=acc_names[i])
#                 plt.xlabel(xlabel)
#                 plt.xticks(alg_names)
#                 plt.ylabel(ylabel)
#                 plt.ylim([-0.1, 1.1])
#                 plt.legend()
#                 plt.savefig(
#                     f"{output_path}/res_{graph_names[graph_number]}_{noises[n]}.png")


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

    # T = [0, 4, 1, 2, 3]
    # (g,n,i,alg,mt,acc)
    # (g,n,i,alg,acc)
    # (g,acc,n,i,alg)
    # T = [0, 4, 3, 1, 2]
    # (g,acc,alg,n,i)

    T = [0, 3, 4, 5, 1, 2]
    # (g,alg,mt,acc,n,i)
    dims = [
        graph_names,
        noises,
        list(range(1, iters+1)),
        [a[3] for a in algs],
        mt_names,
        acc_names
    ]

    time6 = np.expand_dims(time5, axis=-1)

    res, dims = trans(res6, dims, T)
    time, _ = trans(time6, list(range(len(T))), T)
    # (g,alg,mt,acc,n,i)
    time_alg = time.take(indices=[0], axis=2)
    time_m = time.take(indices=range(1, time.shape[2]), axis=2)

    # print(time_alg.shape)
    # print(time_m.shape)
    # print(res.shape)

    # print(len(time))
    if acsq:
        res, dims = squeeze(res, dims, 3)
        time_alg, _ = squeeze(time_alg, [], 3)
        time_m, _ = squeeze(time_m, [], 3)
    # print(len(time))
    if mtsq:
        res, dims = squeeze(res, dims, 2)
        time_alg, _ = squeeze(time_alg, [], 2)
        time_m, _ = squeeze(time_m, [], 2)
    # print(len(time))
    if s_trans is not None:
        res, dims = trans(res, dims, s_trans)  # (g,alg,n,i)
        mock = list(range(len(s_trans)))
        time_alg, _ = trans(time_alg, mock, s_trans)  # (g,alg,n,i)
        time_m, _ = trans(time_m, mock, s_trans)  # (g,alg,n,i)
    # print(len(time))
    save_rec(res, dims, f"{output_path}/acc")
    # print(len(dims))
    # print(time.shape)
    save_rec(time_alg, dims, f"{output_path}/time_alg", plot_type=2)
    save_rec(time_m, dims, f"{output_path}/time_matching", plot_type=2)

    # if save_type == 0:
    #     res5 = np.squeeze(res6, axis=4)

    # if save_type == 1:
    #     sq = 4
    #     res5 = np.squeeze(res6, axis=sq)
    #     res5 = res5.transpose(*T)

    #     del dims[sq]
    #     dims = [dims[i] for i in T]
    # elif save_type == 2:
    #     T = [T[i] for i in [0, 2, 1, 3, 4]]
    #     sq = 5
    #     res5 = np.squeeze(res6, axis=sq)
    #     res5 = res5.transpose(*T)

    #     del dims[sq]
    #     dims = [dims[i] for i in T]
    # elif save_type == 3:
    #     T = [T[i] for i in [3, 1, 2, 0, 4]]
    #     sq = 5
    #     res5 = np.squeeze(res6, axis=sq)
    #     res5 = res5.transpose(*T)

    #     del dims[sq]
    #     dims = [dims[i] for i in T]

    # saveexls(res, filename="accs",
    #          dim1=dims[-4],
    #          dim2=dims[-3],
    #          dim3=dims[-2],
    #          dim4=dims[-1],
    #          )

    # plotrees(np.mean(res, axis=-1), filename="accs",
    #          dim1=dims[-4],
    #          dim2=dims[-3],
    #          dim3=dims[-2],
    #          )

# python workexp.py -l DEBUG with playground full load=[862,862,862]
