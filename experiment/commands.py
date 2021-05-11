from . import ex, save
import numpy as np
import json

# def extract_metric(metric, size):


@ex.command
def e_accs(_run, graphs, noises, iters, algs, acc_names, graph_names, _id=None, squeeze=4, mt_names=[]):

    _id = _run._id if _id is None else _id

    with open(f"runs/{_id}/metrics.json") as f:
        metrics = json.load(f)

    size = (len(graphs), len(noises), iters, 1, 1, 1)

    # accs = [*acc_names, "time"]

    arr = []
    for _, _, mts, algname in algs:
        arr2 = []
        for mt in mts:
            arr3 = []
            for acc in acc_names:
                vals = metrics[f"{algname}.{mt}.{acc}"]["values"]
                vals = np.array(vals).reshape(size)
                arr3.append(vals)
            arr3 = np.concatenate(arr3, axis=5)
            arr2.append(arr3)
        arr2 = np.concatenate(arr2, axis=4)
        arr.append(arr2)
    arr = np.concatenate(arr, axis=3)

    # save.savexls(arr, prefix="accs", acc_names=accs)
    # save.savexls(arr, prefix="accs")
    print(arr.shape)  # g, n, it, alg, mt, acc

    arr = np.squeeze(arr, axis=squeeze)

    if squeeze == 5:
        acc_names = mt_names

    save.saveexls(arr.transpose(0, 4, 1, 2, 3), prefix="accs",
                  dim1=graph_names,
                  dim2=acc_names,
                  dim3=noises,
                  dim4=list(range(1, iters+1)),
                  dim5=[a[3] for a in algs],
                  )

    save.plotrees(np.mean(arr, axis=2).transpose(0, 3, 2, 1), prefix="accs",
                  dim1=graph_names,
                  dim2=acc_names,
                  dim3=[a[3] for a in algs],
                  dim4=noises,
                  )


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
