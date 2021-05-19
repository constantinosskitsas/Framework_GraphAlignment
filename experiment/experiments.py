from . import ex, _algs, _CONE_args, _GRASP_args, _GW_args, _ISO_args, _KLAU_args, _LREA_args, _NET_args, _NSD_args, _REGAL_args, generate as gen
from algorithms import regal, eigenalign, conealign, netalign, NSD, klaus, gwl, grasp2 as grasp, isorank2 as isorank
from networkx import nx


# mprof run workexp.py with playground run=[1,2,3,4,5] iters=2 win

def alggs(tmp):
    alg, args, mtype, algname = _algs[tmp[0]]
    return [
        # (alg, {**args, **update}, mtype, f"{algname}{list(update.values())[0]}") for update in tmp[1]
        (alg, {**args, **update}, mtype, str(list(update.values())[0])) for update in tmp[1]
    ]


@ex.named_config
def scaling():

    # Greedied down
    _algs[0][2][0] = 2
    _CONE_args['window'] = 4
    _algs[1][2][0] = -2
    _algs[2][2][0] = -2
    _algs[3][2][0] = -2
    _algs[4][2][0] = 2
    _algs[5][2][0] = 2
    _algs[6][2][0] = 2

    run = [3, 4, 5]

    iters = 1

    graph_names = [
        # "100",
        # "1000",
        # "10000",
        # "100000",

        '1024',
        '2048',
        '4096',
        '8192',
        '16384',  # 2 ** 14
        '32768',
        # '65536',
        # '131072',
    ]

    graphs = [
        # (nx.powerlaw_cluster_graph, (100, 2, 0.5)),
        # (nx.powerlaw_cluster_graph, (1000, 2, 0.5)),
        # (nx.powerlaw_cluster_graph, (10000, 2, 0.5)),
        # (nx.powerlaw_cluster_graph, (100000, 2, 0.5)),

        (nx.powerlaw_cluster_graph, (1024, 2, 0.5)),
        (nx.powerlaw_cluster_graph, (2048, 2, 0.5)),
        (nx.powerlaw_cluster_graph, (4096, 2, 0.5)),
        (nx.powerlaw_cluster_graph, (8192, 2, 0.5)),
        (nx.powerlaw_cluster_graph, (16384, 2, 0.5)),  # 2 ** 14
        (nx.powerlaw_cluster_graph, (32768, 2, 0.5)),
        # (nx.powerlaw_cluster_graph, (65536, 2, 0.5)),
        # (nx.powerlaw_cluster_graph, (131072, 2, 0.5)),
    ]

    noises = [
        0.05,
    ]

    s_trans = (2, 1, 0, 3)
    xlabel = "powerlaw"


@ex.named_config
def tuning():

    # tmp = [
    #     1,  # CONE
    #     [
    #         {'dim': 128 * i} for i in range(1, 17)
    #     ]
    # ]

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
    ]

    graphs = [
        # (gen.loadnx, ('data/arenas/source.txt',)),
        (gen.loadnx, ('data/facebook/source.txt',)),
        # (gen.loadnx, ('data/CA-AstroPh/source.txt',)),
    ]

    noises = [
        0.01,
        0.03,
        0.05,
    ]

    s_trans = (0, 2, 1, 3)
    xlabel = list(tmp[1][0].keys())[0]


@ex.named_config
def real():

    run = [1, 2, 3, 4, 5, 6]

    iters = 5

    graph_names = [
        "arenas",
        # "facebook",
        # "astro",
    ]

    graphs = [
        (gen.loadnx, ('data/arenas/source.txt',)),
        # with real load=[2-,2-] / iters10 / 0-6
        # (gen.loadnx, ('data/facebook/source.txt',)),
        # with real load=[3-,3-] / iters5 / 0 + 1-6
        # (gen.loadnx, ('data/CA-AstroPh/source.txt',)),
    ]

    noises = [
        0.00,
        0.01,
        0.02,
        0.03,
        0.04,
        0.05,
    ]


@ex.named_config
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
        (gen.loadnx, ('data/arenas/source.txt',)),
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


@ex.named_config
def tuned():
    _CONE_args["dim"] = 512
    _LREA_args["iters"] = 40
    _ISO_args["alpha"] = 0.9
    _ISO_args["lalpha"] = 100000  # full dim


@ex.named_config
def test():

    graph_names = [
        "test1",
        "test2",
    ]

    graphs = [
        # (gen.loadnx, ('data/arenas/source.txt',)),
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
