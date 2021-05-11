from . import ex, _algs


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
def full():

    prep = True

    # lalpha = 1
    # lalpha = None

    # mind = None
    # mind = 1e-8

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

    # mt_all = [
    #     1, 2, 3, 30, -1, -2, -3, -30 ...
    # ]

    # mt_names = [
    #     # "old_douche"
    #     "SNN",
    #     "SSG",
    #     "SJV",
    #     "SJVl",
    #     "CNN",
    #     "CSG",
    #     "CJV",
    #     "CJVl",
    # ]

    # acc_names = ['acc']
    # squeeze = 5

    # xls_type = 2
    # plot_type = 2
