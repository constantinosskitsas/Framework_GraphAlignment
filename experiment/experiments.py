from . import ex, _algs, _CONE_args, _GRASP_args, _GW_args, _ISO_args, _KLAU_args, _LREA_args, _NET_args, _NSD_args, _REGAL_args
from generation import generate as gen
from algorithms import regal, eigenalign, conealign, netalign, NSD, klaus, gwl, grasp2 as grasp, isorank2 as isorank
from networkx import nx
import numpy as np

# mprof run workexp.py with playground run=[1,2,3,4,5] iters=2 win


def alggs(tmp):
    alg, args, mtype, algname = _algs[tmp[0]]
    return [
        # (alg, {**args, **update}, mtype, f"{algname}{list(update.values())[0]}") for update in tmp[1]
        (alg, {**args, **update}, mtype, str(list(update.values())[0])) for update in tmp[1]
    ]


# @ex.named_config
# def scaling():

#     # Greedied down
#     _algs[0][2][0] = 2
#     _CONE_args['window'] = 4
#     _algs[1][2][0] = -2
#     _algs[2][2][0] = -2
#     _algs[3][2][0] = -2
#     _algs[4][2][0] = 2
#     _algs[5][2][0] = 2
#     _algs[6][2][0] = 2

#     run = [3, 4, 5]

#     iters = 1

#     graph_names = [
#         # "100",
#         # "1000",
#         # "10000",
#         # "100000",

#         # '1024',
#         # '2048',
#         # '4096',
#         # '8192',
#         # '16384',  # 2 ** 14
#         '32768',
#         # '65536',
#         # '131072',
#     ]

#     graphs = [
#         # (nx.powerlaw_cluster_graph, (100, 2, 0.5)),
#         # (nx.powerlaw_cluster_graph, (1000, 2, 0.5)),
#         # (nx.powerlaw_cluster_graph, (10000, 2, 0.5)),
#         # (nx.powerlaw_cluster_graph, (100000, 2, 0.5)),

#         # (nx.powerlaw_cluster_graph, (1024, 2, 0.5)),
#         # (nx.powerlaw_cluster_graph, (2048, 2, 0.5)),
#         # (nx.powerlaw_cluster_graph, (4096, 2, 0.5)),
#         # (nx.powerlaw_cluster_graph, (8192, 2, 0.5)),
#         # (nx.powerlaw_cluster_graph, (16384, 2, 0.5)),  # 2 ** 14
#         (nx.powerlaw_cluster_graph, (32768, 2, 0.5)),
#         # (nx.powerlaw_cluster_graph, (65536, 2, 0.5)),
#         # (nx.powerlaw_cluster_graph, (131072, 2, 0.5)),
#     ]

#     noises = [
#         # 0.00,
#         0.05,
#     ]

#     s_trans = (2, 1, 0, 3)
#     xlabel = "powerlaw"

def aaa(vals, dist_type=0):
    g = []
    for val in vals:
        if dist_type == 0:
            dist = np.random.randint(15, 21, val)
        if dist_type == 1:
            dist = nx.utils.powerlaw_sequence(val, 2.5)
            dist = np.array(dist)
            dist = dist.round()
            dist += 1
            dist = dist.tolist()
        if dist_type == 2:
            dist = np.random.normal(10, 1, val)
            # dist = np.random.normal(val, 1, 2**14)
        if dist_type == 3:
            dist = np.random.poisson(lam=10, size=val)

        dist = [round(num) for num in dist]
        usum = sum(dist)
        if usum % 2 == 1:
            max_value = max(dist)
            max_index = dist.index(max_value)
            dist[max_index] = dist[max_index]-1
        G2 = nx.configuration_model(dist, nx.Graph)
        G2.remove_edges_from(nx.selfloop_edges(G2))
        g.append((lambda x: x, (G2,)))
    return g
    # normald = np.random.normal(10, 2, 1000) make it 1 for standard


def ggg(vals):
    return [str(x) for x in vals]


@ex.named_config
def scaling():

    # Greedied down
    _algs[0][2][0] = 2
    # _CONE_args['window'] = 4
    _algs[1][2][0] = -2
    _algs[2][2][0] = -2
    _algs[3][2][0] = -2
    _algs[4][2][0] = 2
    _algs[5][2][0] = 2
    _algs[6][2][0] = 2

    _CONE_args["dim"] = 256
    _GRASP_args["n_eig"] = 256
    _ISO_args["alpha"] = 0.9
    _ISO_args["lalpha"] = 100000  # full dim

    # run = [3, 4, 5]
    run = [1, 2, 3, 4, 5, 6]

    iters = 5

    tmp = [
        # 2**i for i in range(10, 14)
        # 2 ** 15,
        2 ** 16,
        # 2 ** 17,
        # 10, 100, 1000, 10000
    ]

    # graphs = aaa(tmp, dist_type=0)
    # xlabel = "kdist"
    # graphs = aaa(tmp, dist_type=1)
    # xlabel = "powerlaw"
    graphs = aaa(tmp, dist_type=2)
    xlabel = "normal"
    # graphs = aaa(tmp, dist_type=3)
    # xlabel = "poisson"

    graph_names = ggg(tmp)

    # graph_names = [
    #     # "100",
    #     # "1000",
    #     # "10000",
    #     # "100000",

    #     '1024',
    #     '2048',
    #     '4096',
    #     '8192',
    #     # '16384',  # 2 ** 14
    #     # '32768',
    #     # '65536',
    #     # '131072',
    # ]

    # graphs = [
    #     # (nx.powerlaw_cluster_graph, (100, 2, 0.5)),
    #     # (nx.powerlaw_cluster_graph, (1000, 2, 0.5)),
    #     # (nx.powerlaw_cluster_graph, (10000, 2, 0.5)),
    #     # (nx.powerlaw_cluster_graph, (100000, 2, 0.5)),

    #     (nx.powerlaw_cluster_graph, (1024, 2, 0.5)),
    #     (nx.powerlaw_cluster_graph, (2048, 2, 0.5)),
    #     (nx.powerlaw_cluster_graph, (4096, 2, 0.5)),
    #     (nx.powerlaw_cluster_graph, (8192, 2, 0.5)),
    #     # (nx.powerlaw_cluster_graph, (16384, 2, 0.5)),  # 2 ** 14
    #     # (nx.powerlaw_cluster_graph, (32768, 2, 0.5)),
    #     # (nx.gnp_random_graph, (32768, 0.0003)),
    #     # (nx.powerlaw_cluster_graph, (65536, 2, 0.5)),
    #     # (nx.powerlaw_cluster_graph, (131072, 2, 0.5)),
    # ]

    noises = [
        # 0.00,
        0.05,
    ]

    s_trans = (2, 1, 0, 3)
    # xlabel = "k_normal"


@ex.named_config
def tuning():

    # tmp = [
    #     1,  # CONE
    #     [
    #         {'dim': 128 * i} for i in range(1, 17)
    #     ]
    # ]

    # tmp = [
    #     2,  # grasp
    #     [
    #         {'n_eig': x} for x in [128, 512, 1024]
    #     ]
    # ]
    # _algs[2][2][0] = -2

    # tmp = [
    #     3,  # REGAL
    #     [
    #         {'untillayer': x} for x in range(1, 6)
    #     ]
    # ]

    # tmp = [
    #     4,  # LREA
    #     [
    #         {'iters': 8 * i} for i in range(1, 9)
    #     ]
    # ]

    # tmp = [
    #     5,  # NSD
    #     [
    #         {'iters': x} for x in [15, 20, 25, 30, 35, 40]
    #     ]
    # ]

    tmp = [
        6,  # ISO
        [
            # {'lalpha': x} for x in [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 99999]
            # {'alpha': x} for x in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
            {'alpha': x} for x in [0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 0.999, 0.9999]
        ]
    ]

    # _ISO_args["alpha"] = 0.8
    _ISO_args["lalpha"] = 40
    # _ISO_args["weighted"] = False

    _algs[:] = alggs(tmp)

    run = list(range(len(tmp[1])))

    iters = 5

    graph_names = [
        # "arenas",
        "facebook",
        # "astro",
        # "gnp"
    ]

    graphs = [
        # (gen.loadnx, ('data/arenas.txt',)),
        (gen.loadnx, ('data/facebook.txt',)),
        # (gen.loadnx, ('data/CA-AstroPh.txt',)),
        # (nx.gnp_random_graph, (2**15, 0.0003)),
    ]

    noises = [
        0.01,
        0.03,
        0.05,
    ]

    s_trans = (0, 2, 1, 3)
    xlabel = list(tmp[1][0].keys())[0]


def namess(tmp):
    return [name[-15:] for name in tmp[1]]


def graphss(tmp):
    return [
        (lambda x:x, [[
            tmp[0],
            target,
            None
        ]]) for target in tmp[1]
    ]


@ex.named_config
def real_noise():

    tmp = [
        "data/real world/contacts-prox-high-school-2013/contacts-prox-high-school-2013_100.txt",
        [
            f"data/real world/contacts-prox-high-school-2013/contacts-prox-high-school-2013_{i}.txt" for i in [
                99, 95, 90, 80]
        ]
    ]
    xlabel = "high-school-2013"

    # tmp = [
    #     "data/real world/mamalia-voles-plj-trapping/mammalia-voles-plj-trapping_100.txt",
    #     [
    #         f"data/real world/mamalia-voles-plj-trapping/mammalia-voles-plj-trapping_{i}.txt" for i in [
    #             99, 95, 90, 80]
    #     ]
    # ]
    # xlabel = "mammalia-voles"

    # tmp = [
    #     "data/real world/MultiMagna/yeast0_Y2H1.txt",
    #     [
    #         f"data/real world/MultiMagna/yeast{i}_Y2H1.txt" for i in [
    #             5, 10, 15, 20, 25]
    #     ]
    # ]
    # xlabel = "yeast_Y2H1"

    graph_names = namess(tmp)

    graphs = graphss(tmp)

    iters = 2

    noises = [
        1.0,
    ]

    s_trans = (2, 1, 0, 3)

    # (g,alg,acc,n,i)
    # s_trans = (3, 1, 2, 0, 4)


def rgraphs(gnames):
    return [
        (gen.loadnx, (f"data/{name}.txt",)) for name in gnames
    ]


@ ex.named_config
def real():

    run = [1, 2, 3, 4, 5, 6]

    iters = 5

    graph_names = [             # n     / e
        "ca-netscience",        # 379   / 914   / connected
        "bio-celegans",         # 453   / 2k    / connected
        "in-arenas",            # 1.1k  / 5.4k  / connected
        "inf-euroroad",         # 1.2K  / 1.4K  / disc - 200
        "inf-power",            # 4.9K  / 6.6K  / connected
        "ca-GrQc",              # 4.2k  / 13.4K / connected - (5.2k  / 14.5K)?
        "bio-dmela",            # 7.4k  / 25.6k / connected
        "CA-AstroPh",           # 18k   / 195k  / connected

        "soc-hamsterster",      # 2.4K  / 16.6K / disc - 400
        "socfb-Bowdoin47",      # 2.3K  / 84.4K / disc - only 2
        "socfb-Hamilton46",     # 2.3K  / 96.4K / disc - only 2
        "socfb-Haverford76",    # 1.4K  / 59.6K / connected
        "socfb-Swarthmore42",   # 1.7K  / 61.1K / disc - only 2
        "soc-facebook",         # 4k    / 87k   / connected

        # "ca-Erdos992",          # 6.1K  / 7.5K  / disc - 100 + 1k disc nodes
    ]

    graphs = rgraphs(graph_names)

    # graphs = [
    #     (gen.loadnx, ('data/arenas.txt',)),
    #     # with real load=[2-,2-] / iters10 / 0-6
    #     (gen.loadnx, ('data/facebook.txt',)),
    #     # with real load=[3-,3-] / iters5 / 0 + 1-6
    #     # (gen.loadnx, ('data/CA-AstroPh.txt',)),

    #     (gen.loadnx, ('data/bio-celegans.mtx',)),
    #     (gen.loadnx, ('data/bio-dmela.mtx',)),
    #     (gen.loadnx, ('data/ca-Erdos992.mtx',)),
    #     (gen.loadnx, ('data/ca-GrQc.mtx',)),
    #     (gen.loadnx, ('data/ca-netscience.mtx',)),
    #     (gen.loadnx, ('data/inf-euroroad.edges',)),
    #     (gen.loadnx, ('data/inf-power.mtx',)),
    #     (gen.loadnx, ('data/soc-hamsterster.edges',)),
    #     (gen.loadnx, ('data/socfb-Bowdoin47.mtx',)),
    #     (gen.loadnx, ('data/socfb-Hamilton46.mtx',)),
    #     (gen.loadnx, ('data/socfb-Haverford76.mtx',)),
    #     (gen.loadnx, ('data/socfb-Swarthmore42.mtx',)),
    # ]

    noises = [
        # 0.00,
        # 0.01,
        # 0.02,
        # 0.03,
        # 0.04,
        # 0.05,

        0.00,
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
    ]


@ ex.named_config
def arenasish():

    # use with 'mall'

    iters = 10

    graph_names = [
        "arenas",
        "powerlaw",
        "nw_str",
        "watts_str",
        "gnp",
        "barabasi",
    ]

    graphs = [
        # with arenasish load=[1-,1-]
        # 91-
        (gen.loadnx, ('data/arenas.txt',)),
        (nx.powerlaw_cluster_graph, (1133, 5, 0.5)),
        # 92-0
        (nx.newman_watts_strogatz_graph, (1133, 7, 0.5)),
        (nx.watts_strogatz_graph, (1133, 10, 0.5)),
        # 92-1
        (nx.gnp_random_graph, (1133, 0.009)),
        (nx.barabasi_albert_graph, (1133, 5)),
    ]

    noises = [
        0.00,
        0.01,
        0.02,
        0.03,
        0.04,
        0.05,
    ]


@ ex.named_config
def tuned():
    _CONE_args["dim"] = 512
    # _CONE_args["dim"] = 256
    # _GRASP_args["n_eig"] = 256
    _LREA_args["iters"] = 40
    _ISO_args["alpha"] = 0.9
    _ISO_args["lalpha"] = 100000  # full dim
    # _ISO_args["lalpha"] = 25


@ ex.named_config
def test():

    graph_names = [
        "test1",
        "test2",
    ]

    graphs = [
        # (gen.loadnx, ('data/arenas.txt',)),
        (nx.gnp_random_graph, (50, 0.5)),
        (nx.barabasi_albert_graph, (50, 3)),
    ]

    run = [1, 3, 5]

    iters = 4

    noises = [
        0.00,
        0.01,
        0.02,
        0.03,
        0.04,
    ]
