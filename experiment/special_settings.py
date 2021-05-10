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


def jv_to_sps(val):
    if val == [4]:
        return [3]
    if val == [-4]:
        return [-3]
    if val == [40]:
        return [30]
    if val == [-40]:
        return [-30]
    return val


@ex.named_config
def win():

    algs = [
        (alg, args, jv_to_sps(mtype), algname) for alg, args, mtype, algname in _algs
    ]


# @ex.named_config
# def fast():

#     GW_args = {
#         'opt_dict': {
#             'epochs': 1,
#             'outer_iteration': 40,
#             'sgd_iteration': 30,
#         },
#         'hyperpara_dict': {
#             'dimension': 5
#         },
#         'max_cpu': 0
#     }

#     GRASP_args = {
#         'n_eig': 50,
#         'k': 5
#     }

#     CONE_args = {
#         'dim': 16
#     }

#     run = [
#         0,      # gwl
#         1,      # conealign,
#         2,      # grasp,
#         3,      # regal,
#     ]

#     mnc = False


@ex.named_config
def full():

    prep = True

    # lalpha = 1
    # lalpha = None

    # mind = None
    # mind = 1e-8

    run = [0, 1, 2, 3, 4, 5, 6, 7, 8]
