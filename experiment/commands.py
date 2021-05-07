from . import ex, save
import numpy as np
import json

# def extract_metric(metric, size):


@ex.command
def extract_metric():

    metric = [
        "C:\\Users\\KAROLEK\\Desktop\\Thesis\\189\\metrics.json",
        # "algorithms.LREA.eigenalign.alg",
        "algorithms.CONE.conealign.alg",
        "values"
    ]

    size = (1, 3, 10, 6, 1)
    with open(metric[0]) as f:
        vals = json.load(f)
        for m in metric[1:]:
            vals = vals[m]
        vals = np.array(vals).reshape(size)

    save.savexls(vals, "results", np.arange(vals.shape[-2]), None, None, None)


# ["C:\\Users\\KAROLEK\\Desktop\\Thesis\\188\\metrics.json","algorithms.LREA.eigenalign.alg","values"]
# (3,10,8)
