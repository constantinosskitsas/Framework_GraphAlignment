from . import ex, save
import numpy as np
import json

# def extract_metric(metric, size):


@ex.command
def e_accs(_run, graphs, noises, iters, algs, acc_names, _id=None):

    _id = _run._id if _id is None else _id

    with open(f"runs/{_id}/metrics.json") as f:
        metrics = json.load(f)

    size = (len(graphs), len(noises), iters, 1, 1)

    # accs = [*acc_names, "time"]

    arr = []
    for _, _, mts, algname in algs:
        arr2 = []
        # for acc in accs:
        for acc in acc_names:
            vals = metrics[f"{algname}.{mts[0]}.{acc}"]["values"]
            vals = np.array(vals).reshape(size)
            arr2.append(vals)
        arr2 = np.concatenate(arr2, axis=4)
        arr.append(arr2)

    arr = np.concatenate(arr, axis=3)

    # save.savexls(arr, prefix="accs", acc_names=accs)
    save.savexls(arr, prefix="accs")


@ex.command
def e_time(_run, graphs, noises, iters, algs, acc_names, _id=None):

    _id = _run._id if _id is None else _id

    with open(f"runs/{_id}/metrics.json") as f:
        metrics = json.load(f)

    size = (len(graphs), len(noises), iters, 1, 1)

    arr = []
    for _, _, _, algname in algs:
        vals = metrics[f"{algname}.time"]["values"]
        vals = np.array(vals).reshape(size)
        arr.append(vals)

    arr = np.concatenate(arr, axis=3)

    save.savexls(arr, prefix="time")


# ["C:\\Users\\KAROLEK\\Desktop\\Thesis\\188\\metrics.json","algorithms.LREA.eigenalign.alg","values"]
# (3,10,8)
