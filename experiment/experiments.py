from . import ex, _CONE_args, _GRASP_args, _GW_args, _ISO_args, _KLAU_args, _LREA_args, _NET_args, _NSD_args, _REGAL_args
from algorithms import regal, eigenalign, conealign, netalign, NSD, klaus, gwl, grasp2 as grasp, isorank2 as isorank
from networkx import nx


def alggs(tmp):
    return [
        (tmp[0], {**tmp[1], **update}, tmp[2]) for update in tmp[3]
    ]


@ex.named_config
def exp4():

    # tmp = [
    #     conealign,
    #     _CONE_args,
    #     -4,
    #     [
    #         {'dim': 64},
    #         {'dim': 128},
    #         {'dim': 256},
    #         {'dim': 512},
    #         {'dim': 768},
    #         {'dim': 1024},
    #     ]
    # ]

    # tmp = [
    #     grasp,
    #     _GRASP_args,
    #     -4,
    #     [
    #         {'k': 15},
    #         {'k': 20},
    #         {'k': 25},
    #         {'q': 90},
    #         {'q': 100},
    #         {'q': 110},
    #     ]
    # ]

    tmp = [
        regal,
        _REGAL_args,
        -4,
        [
            {'untillayer': 1},
            {'untillayer': 2},
            {'untillayer': 3},
            {'untillayer': 4},
            {'untillayer': 5},
        ]
    ]

    # tmp = [
    #     eigenalign,
    #     _LREA_args,
    #     4,
    #     [
    #         {'iters': 5},
    #         {'iters': 8},
    #         {'iters': 11},
    #         {'iters': 14},
    #         {'iters': 17},
    #         {'iters': 20},
    #     ]
    # ]

    # tmp = [
    #     NSD,
    #     _NSD_args,
    #     40,
    #     [
    #         {'iters': 15},
    #         {'iters': 20},
    #         {'iters': 25},
    #         {'iters': 30},
    #         {'iters': 35},
    #         {'iters': 40},
    #     ]
    # ]

    algs = alggs(tmp)

    try:
        run
    except NameError:
        run = list(range(len(tmp[3])))

    try:
        xlabel
    except NameError:
        xlabel = list(tmp[3][0].keys())[0]

    try:
        alg_names
    except NameError:
        alg_names = [list(d.values())[0] for d in tmp[3]]

    iters = 10

    graph_names = [
        "arenas005",
        "facebook005",
    ]

    graphs = [
        (lambda x: x, ('data/arenas/source.txt',)),
        (lambda x: x, ('data/facebook/source.txt',)),
    ]

    noises = [
        0.05,
    ]

    plot_type = 3


@ex.named_config
def exp3():

    CONE_args = {
        "dim": 512
    }

    # LREA_args = {
    #     'iters': 40
    # }

    # NSD_args = {
    #     'iters': 30
    # }

    run = [1, 2, 3, 4, 5]

    iters = 10

    graph_names = [
        "facebook",
    ]

    graphs = [
        (lambda x: x, ('data/facebook/source.txt',)),
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
def exp2():

    iters = 10

    graph_names = [
        "nw_str",
        "watts_str",
        "gnp",
        "barabasi",
        "powerlaw"
    ]

    graphs = [
        (nx.newman_watts_strogatz_graph, (1133, 7, 0.5)),
        (nx.watts_strogatz_graph, (1133, 10, 0.5)),
        (nx.gnp_random_graph, (1133, 0.009)),
        (nx.barabasi_albert_graph, (1133, 5)),
        # (nx.powerlaw_cluster_graph, (1133, 5, 0.5)),
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
def mall():
    mall = True
    mtypes = [1, 2, 3, 30, 4, 40, -1, -2, -3, -30, -4, -40]
    acc_names = [
        # "old_douche"
        "SNN",
        "SSG",
        "SLS",
        "SLSl",
        "SJV",
        "SJVl",
        "CNN",
        "CSG",
        "CLS",
        "CLSl",
        "CJV",
        "CJVl",
    ]

    xls_type = 2
    plot_type = 2


@ex.named_config
def exp1():

    iters = 10

    graph_names = [
        "arenas",
        "powerlaw"
    ]

    graphs = [
        (lambda x: x, ('data/arenas/source.txt',)),
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
