
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import collections
import scipy.sparse as sps
import sys
import os
try:
    import lapjv
except:
    pass

from data import similarities_preprocess
from algorithms import bipartitewrapper as bmw


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


def e_to_G(e, n):
    # n = np.amax(e) + 1
    nedges = e.shape[0]
    G = sps.csr_matrix((np.ones(nedges), e.T), shape=(n, n), dtype=int)
    G += G.T
    G.data = G.data.clip(0, 1)
    return G


def load_as_nx(path):
    G_e = np.loadtxt(path, int)
    G = nx.Graph(G_e.tolist())
    return np.array(G.edges)


def refill_e(edges, n, amount):
    if amount == 0:
        return edges
    # print(edges)
    # ee = np.sort(edges).tolist()
    ee = {tuple(row) for row in np.sort(edges).tolist()}
    new_e = []
    check = 0
    while len(new_e) < amount:
        _e = np.random.randint(n, size=2)
        # _ee = np.sort(_e).tolist()
        _ee = tuple(np.sort(_e).tolist())
        check += 1
        if not(_ee in ee) and _e[0] != _e[1]:
            # ee.append(_ee)
            ee.add(_ee)
            new_e.append(_e)
            check = 0
            # print(f"refill - {len(new_e)}/{amount}")
        if check % 1000 == 999:
            print(f"refill - {check + 1} times in a row fail")
    # print(new_e)
    return np.append(edges, new_e, axis=0)


def remove_e(edges, noise, no_disc):
    if no_disc:
        bin_count = np.bincount(edges.flatten())
        rows_to_delete = []
        for i, edge in enumerate(edges):
            if np.random.sample(1)[0] < noise:
                e, f = edge
                if bin_count[e] > 1 and bin_count[f] > 1:
                    bin_count[e] -= 1
                    bin_count[f] -= 1
                    rows_to_delete.append(i)
        edges = np.delete(edges, rows_to_delete, axis=0)
    else:
        edges = edges[np.random.sample(edges.shape[0]) >= noise]
    return edges


def generate_graphs(G, source_noise=0.00, target_noise=0.00, refill=False, no_disc=False):

    if isinstance(G, dict):
        dataset = G['dataset']
        edges = G['edges']
        noise_level = G['noise_level']

        source = f"data/{dataset}/source.txt"
        target = f"data/{dataset}/noise_level_{noise_level}/edges_{edges}.txt"
        grand_truth = f"data/{dataset}/noise_level_{noise_level}/gt_{edges}.txt"

        Src_e = load_as_nx(source)
        Tar_e = load_as_nx(target)
        gt_e = np.loadtxt(grand_truth, int).T

        # Src = e_to_G(Src_e)
        # Tar = e_to_G(Tar_e)

        Gt = (
            gt_e[:, gt_e[1].argsort()][0],
            gt_e[:, gt_e[0].argsort()][1]
        )

        return Src_e, Tar_e, Gt
    elif isinstance(G, str):
        Src_e = load_as_nx(G)
    elif isinstance(G, nx.Graph):
        Src_e = np.array(G.edges)
    else:
        return sps.csr_matrix([]), sps.csr_matrix([]), (np.empty(1), np.empty(1))

    n = np.amax(Src_e) + 1
    nedges = Src_e.shape[0]

    gt_e = np.array((
        np.arange(n),
        np.random.permutation(n)
    ))

    Gt = (
        gt_e[:, gt_e[1].argsort()][0],
        gt_e[:, gt_e[0].argsort()][1]
    )

    Tar_e = Gt[0][Src_e]

    Src_e = remove_e(Src_e, source_noise, no_disc)
    Tar_e = remove_e(Tar_e, target_noise, no_disc)

    # if no_disc:
    #     Src_bin = np.bincount(Src_e.flatten())
    #     rows_to_delete = []
    #     for i, edge in enumerate(Src_e):
    #         if np.random.sample(1)[0] < source_noise:
    #             e, f = edge
    #             if Src_bin[e] > 1 and Src_bin[f] > 1:
    #                 Src_bin[e] -= 1
    #                 Src_bin[f] -= 1
    #                 rows_to_delete.append(i)
    #     Src_e = np.delete(Src_e, rows_to_delete, axis=0)

    #     Tar_bin = np.bincount(Tar_e.flatten())
    #     rows_to_delete = []
    #     for i, edge in enumerate(Tar_e):
    #         if np.random.sample(1)[0] < target_noise:
    #             e, f = edge
    #             if Tar_bin[e] > 1 and Tar_bin[f] > 1:
    #                 Tar_bin[e] -= 1
    #                 Tar_bin[f] -= 1
    #                 rows_to_delete.append(i)
    #     Tar_e = np.delete(Tar_e, rows_to_delete, axis=0)

    # else:
    #     Src_e = Src_e[np.random.sample(nedges) >= source_noise]
    #     Tar_e = Tar_e[np.random.sample(nedges) >= target_noise]

    if refill:
        Src_e = refill_e(Src_e, n, nedges - Src_e.shape[0])
        Tar_e = refill_e(Tar_e, n, nedges - Tar_e.shape[0])

    return Src_e, Tar_e,  Gt


def load_G(path):
    G_e = np.loadtxt(G, int)
    return nx.Graph(G_e.tolist())


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


# def plot(cx, filename):
#     connects = np.zeros(cx.shape)
#     for row, col in zip(cx.row, cx.col):
#         connects[row, col] += 1
#     plt.imshow(connects)
#     plt.savefig(f'results/{filename}.png')
#     plt.close('all')


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


# def plotGs(left, right, circular=False):
#     G_left = nx.Graph(left)
#     G_right = nx.Graph(right)
#     if circular:
#         plt.subplot(121)
#         nx.draw(G_left, pos=nx.circular_layout(G_left),
#                 node_color='r', edge_color='b')
#         plt.subplot(122)
#         nx.draw(G_right, pos=nx.circular_layout(G_right),
#                 node_color='r', edge_color='b')
#     else:
#         plt.subplot(121)
#         nx.draw(G_left)
#         plt.subplot(122)
#         nx.draw(G_right)

#     G = G_left

#     degree_sequence = sorted([d for n, d in G.degree()],
#                              reverse=True)  # degree sequence
#     degreeCount = collections.Counter(degree_sequence)
#     deg, cnt = zip(*degreeCount.items())

#     fig, ax = plt.subplots()
#     plt.bar(deg, cnt, width=0.80, color="b")

#     plt.title("Degree Histogram")
#     plt.ylabel("Count")
#     plt.xlabel("Degree")
#     ax.set_xticks([d + 0.4 for d in deg])
#     ax.set_xticklabels(deg)

#     plt.show()


def colmax(matrix):
    ma = np.arange(matrix.shape[0])
    mb = matrix.argmax(1).A1
    return ma, mb


def colmin(matrix):
    ma = np.arange(matrix.shape[0])
    mb = matrix.argmin(1).A1
    return ma, mb


def superfast(l2, asc=True):
    print(f"superfast: init")
    l2 = l2.A
    n = np.shape(l2)[0]
    ma = np.zeros(n, int)
    mb = np.zeros(n, int)
    rows = set()
    cols = set()
    vals = np.argsort(l2, axis=None)
    vals = vals if asc else vals[::-1]
    i = 0
    for x, y in zip(*np.unravel_index(vals, l2.shape)):
        if x in rows or y in cols:
            continue
        print(f"superfast: {i}/{n}")
        i += 1
        ma[x] = x
        mb[x] = y

        rows.add(x)
        cols.add(y)
    return ma, mb


def jv(dist):
    # print('hungarian_matching: calculating distance matrix')

    # dist = sci.spatial.distance_matrix(G1_emb.T, G2_emb.T)
    n = np.shape(dist)[0]
    # print(np.shape(dist))
    # print('hungarian_matching: calculating matching')
    cols, rows, _ = lapjv.lapjv(dist)
    # print(cols)
    # print(rows)
    matching = np.c_[rows, np.linspace(0, n-1, n).astype(int)]
    # print(matching)
    matching = matching[matching[:, 0].argsort()]
    # print(matching)
    return matching.astype(int).T


def S3(A, B, ma, mb):
    A1 = np.sum(A, 0)
    B1 = np.sum(B, 0)
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
    # print(EdA1)
    # print(EdB1)
    # print(Ce)
    res = Ce/div
    return res


def ICorS3GT(A, B, ma, mb, gmb, IC):
    A1 = np.sum(A, 0)
    B1 = np.sum(B, 0)
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


# def get_counterpart(alignment_matrix):
#     counterpart_dict = {}

#     if not sps.issparse(alignment_matrix):
#         sorted_indices = np.argsort(alignment_matrix)

#     n_nodes = alignment_matrix.shape[0]
#     for node_index in range(n_nodes):

#         if sps.issparse(alignment_matrix):
#             row, possible_alignments, possible_values = sps.find(
#                 alignment_matrix[node_index])
#             node_sorted_indices = possible_alignments[possible_values.argsort(
#             )]
#         else:
#             node_sorted_indices = sorted_indices[node_index]
#         counterpart = node_sorted_indices[-1]
#         counterpart_dict[node_index] = counterpart
#     return counterpart_dict


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
