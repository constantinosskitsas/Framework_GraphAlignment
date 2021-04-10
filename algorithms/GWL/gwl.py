from .model.GromovWassersteinLearning import GromovWassersteinLearning
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import scipy.sparse as sps


def main(data, opt_dict, hyperpara_dict, lr, gamma):

    Src = data['Src']
    Tar = data['Tar']

    Se = np.array(sps.find(Src)[:2]).T
    Te = np.array(sps.find(Tar)[:2]).T

    data = {
        'src_index': {float(i): i for i in range(np.amax(Se) + 1)},
        'src_interactions': Se.tolist(),
        'tar_index': {float(i): i for i in range(np.amax(Te) + 1)},
        'tar_interactions': Te.tolist(),
        'mutual_interactions': None
    }

    hyperpara_dictt = {
        'src_number': len(data['src_index']),
        'tar_number': len(data['tar_index']),
        **hyperpara_dict
    }

    gwd_model = GromovWassersteinLearning(hyperpara_dictt)

    # initialize optimizer
    optimizer = optim.Adam(gwd_model.gwl_model.parameters(), lr=lr)

    scheduler = lr_scheduler.ExponentialLR(
        optimizer, gamma=gamma) if gamma else None

    # Gromov-Wasserstein learning
    gwd_model.train_without_prior(
        data, optimizer, opt_dict, scheduler=scheduler)
    cost12 = gwd_model.getCostm()
    # gwd_model.evaluation_recommendation1()
    return gwd_model.trans, cost12
