from .model.GromovWassersteinLearning import GromovWassersteinLearning
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import scipy.sparse as sps


def main(Tar, Src, opt_dict):

    Se = np.array(sps.find(Src)[:2]).T
    Te = np.array(sps.find(Tar)[:2]).T

    data = {
        'src_index': {float(i): i for i in range(np.amax(Se) + 1)},
        'src_interactions': Se.tolist(),
        'tar_index': {float(i): i for i in range(np.amax(Te) + 1)},
        'tar_interactions': Te.tolist(),
        'mutual_interactions': None
    }

    hyperpara_dict = {
        'src_number': len(data['src_index']),
        'tar_number': len(data['tar_index']),
        'dimension': 100,
        'loss_type': 'L2',
        'cost_type': 'cosine',
        'ot_method': 'proximal'
    }

    gwd_model = GromovWassersteinLearning(hyperpara_dict)

    # initialize optimizer
    optimizer = optim.Adam(gwd_model.gwl_model.parameters(), lr=1e-3)

    # scheduler = lr_scheduler.ExponentialLR(
    #     optimizer, gamma=0.8)

    # Gromov-Wasserstein learning
    gwd_model.train_without_prior(data, optimizer, opt_dict, scheduler=None)

    return gwd_model.trans.T
