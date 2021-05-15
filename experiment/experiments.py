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

    iters = 10

    graph_names = [
        "pl10",
        "pl100",
        "pl1000",
        # "pl10000",
    ]

    graphs = [
        (nx.powerlaw_cluster_graph, (10, 2, 0.5)),
        (nx.powerlaw_cluster_graph, (100, 2, 0.5)),
        (nx.powerlaw_cluster_graph, (1000, 2, 0.5)),
        # (nx.powerlaw_cluster_graph, (10000, 2, 0.5)),
    ]

    noises = [
        0.05,
    ]

    s_trans = (2, 1, 0, 3)
    xlabel = "powerlaw"


@ex.named_config
def tuning():

    # tmp = [
    #     1,
    #     [
    #         {'dim': 128},  # arenas
    #         {'dim': 128 * 2},
    #         {'dim': 128 * 3},
    #         {'dim': 128 * 4},
    #         {'dim': 128 * 5},
    #         {'dim': 128 * 6},  # facebook
    #         {'dim': 128 * 7},
    #         {'dim': 128 * 8},
    #         {'dim': 128 * 9},
    #         {'dim': 128 * 10},
    #         {'dim': 128 * 11},
    #         {'dim': 128 * 12},
    #         {'dim': 128 * 13},
    #         {'dim': 128 * 14},
    #         {'dim': 128 * 15},
    #         {'dim': 128 * 16},  # astro
    #     ]
    # ]

    # tmp = [
    #     3,
    #     [
    #         {'untillayer': 1},
    #         {'untillayer': 2},
    #         {'untillayer': 3},
    #         {'untillayer': 4},
    #         {'untillayer': 5},
    #     ]
    # ]

    # tmp = [
    #     4,
    #     [
    #         {'iters': 8},
    #         {'iters': 8 * 2},
    #         {'iters': 8 * 3},
    #         {'iters': 8 * 4},
    #         {'iters': 8 * 5},
    #         {'iters': 8 * 6},
    #         {'iters': 8 * 7},
    #         {'iters': 8 * 8},
    #     ]
    # ]

    # tmp = [
    #     5,
    #     [
    #         {'iters': 15},
    #         {'iters': 20},
    #         {'iters': 25},
    #         {'iters': 30},
    #         {'iters': 35},
    #         {'iters': 40},
    #     ]
    # ]

    tmp = [
        6,
        [
            {'lalpha': 1},
            {'lalpha': 5},
            {'lalpha': 10},
            {'lalpha': 15},
            {'lalpha': 20},
            {'lalpha': 25},
            {'lalpha': 30},
            {'lalpha': 35},
            {'lalpha': 40},
            {'lalpha': 45},
            {'lalpha': 50},
            {'lalpha': 99999},
            # {'lalpha': 1, "weighted": False},
            # {'lalpha': 5, "weighted": False},
            # {'lalpha': 10, "weighted": False},
            # {'lalpha': 15, "weighted": False},
            # {'lalpha': 20, "weighted": False},
            # {'lalpha': 25, "weighted": False},
            # {'lalpha': 30, "weighted": False},
            # {'lalpha': 35, "weighted": False},
            # {'lalpha': 40, "weighted": False},
            # {'lalpha': 45, "weighted": False},
            # {'lalpha': 50, "weighted": False},
            # {'lalpha': 99999, "weighted": False},
        ]
    ]

    # _ISO_args["alpha"] = 0.8

    _algs[:] = alggs(tmp)

    run = list(range(len(tmp[1])))

    iters = 10

    graph_names = [
        "arenas",
        # "facebook",
        # "astro",
    ]

    graphs = [
        (gen.loadnx, ('data/arenas/source.txt',)),
        # (gen.loadnx, ('data/facebook/source.txt',)),
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

    _CONE_args["dim"] = 512

    _LREA_args["iters"] = 40

    # _algs[1][1]["dim"] = 512

    # _algs[4][1]["iters"] = 40

    run = [1, 2, 3, 4, 5]

    iters = 10

    graph_names = [
        "arenas",
        # "facebook",
        # "astro",
    ]

    graphs = [
        (gen.loadnx, ('data/arenas/source.txt',)),
        # (gen.loadnx, ('data/facebook/source.txt',)),
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
        "nw_str",
        "watts_str",
        "gnp",
        "barabasi",
        "powerlaw"
    ]

    graphs = [
        (gen.loadnx, ('data/arenas/source.txt',)),
        (nx.newman_watts_strogatz_graph, (1133, 7, 0.5)),
        (nx.watts_strogatz_graph, (1133, 10, 0.5)),
        (nx.gnp_random_graph, (1133, 0.009)),
        (nx.barabasi_albert_graph, (1133, 5)),
        (nx.powerlaw_cluster_graph, (1133, 5, 0.5)),
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
