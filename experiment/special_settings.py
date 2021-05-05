from . import ex


@ex.named_config
def gwcost():
    GW_args = {
        'opt_dict': {
            'epochs': 10
        }
    }

    # GW_mtype = -4


@ex.named_config
def debug():

    verbose = True
    save = True
    plot = [True, True]


@ex.named_config
def win():

    GW_mtype = 3
    CONE_mtype = -3
    GRASP_mtype = -3
    REGAL_mtype = -3
    LREA_mtype = 3
    NSD_mtype = 30


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


@ex.named_config
def full():

    ISO_args = {
        'alpha': 0.6
    }

    prep = True
    # lalpha = 1
    # mind = None

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
