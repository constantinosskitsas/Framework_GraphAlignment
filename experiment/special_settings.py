from . import ex


# @ex.named_config
# def gwcost():
#     GW_args = {
#         'opt_dict': {
#             'epochs': 10
#         }
#     }

#     # GW_mtype = -4


@ex.named_config
def debug():

    verbose = True
    save = True
    plot = [True, True]


@ex.named_config
def prep():

    prep = True

    # lalpha = 1
    # lalpha = None

    # mind = None
    # mind = 1e-8


@ex.named_config
def rall():

    run = [0, 1, 2, 3, 4, 5, 6, 7, 8]


@ex.named_config
def accall():
    accs = [
        0,
        1,
        2,
        3,
        4,
    ]


@ex.named_config
def mall():

    mall = True

    # save_type = 2

    mt_names = [
        # "old_douche"
        "SNN",
        "SSG",
        "SJV",
        "SJVl",
        "CNN",
        "CSG",
        "CJV",
        "CJVl",
    ]
