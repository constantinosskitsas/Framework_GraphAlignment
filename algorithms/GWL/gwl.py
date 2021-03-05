from .dev import util
from .model.GromovWassersteinLearning import GromovWassersteinLearning
import numpy as np
import torch.optim as optim
from torch.optim import lr_scheduler
import torch
import matplotlib.pyplot as plt


def main(data):
    data_name = 'data'
    result_folder = 'results'
    c = 'cosine'
    m = 'proximal'

    data_mc3 = data

    connects = np.zeros(
        (len(data_mc3['src_index']), len(data_mc3['src_index'])))
    for item in data_mc3['src_interactions']:
        connects[item[0], item[1]] += 1
    plt.imshow(connects)
    plt.savefig('{}/{}_src.png'.format(result_folder, data_name))
    plt.close('all')

    connects = np.zeros(
        (len(data_mc3['tar_index']), len(data_mc3['tar_index'])))
    for item in data_mc3['tar_interactions']:
        connects[item[0], item[1]] += 1
    plt.imshow(connects)
    plt.savefig('{}/{}_tar.png'.format(result_folder, data_name))
    plt.close('all')

    opt_dict = {'epochs': 5,
                'batch_size': 10000,
                'use_cuda': False,
                'strategy': 'soft',
                'beta': 1e-1,
                'outer_iteration': 400,
                'inner_iteration': 1,
                'sgd_iteration': 300,
                'prior': False,
                'prefix': result_folder,
                'display': True}

    hyperpara_dict = {'src_number': len(data_mc3['src_index']),
                      'tar_number': len(data_mc3['tar_index']),
                      'dimension': 20,
                      'loss_type': 'L2',
                      'cost_type': c,
                      'ot_method': m}

    gwd_model = GromovWassersteinLearning(hyperpara_dict)

    # initialize optimizer
    optimizer = optim.Adam(
        gwd_model.gwl_model.parameters(), lr=1e-3)
    scheduler = lr_scheduler.ExponentialLR(
        optimizer, gamma=0.8)

    # print(gwd_model.obtain_embedding(
    #     opt_dict, torch.LongTensor([0, 1]), 0))

    # Gromov-Wasserstein learning
    gwd_model.train_without_prior(
        data_mc3, optimizer, opt_dict, scheduler=None)
    # print(data_mc3)
    # print(optimizer)
    # print(opt_dict)
    # save model
    gwd_model.save_model(
        '{}/model_{}_{}_{}.pt'.format(result_folder, data_name, m, c))
    gwd_model.save_matching(
        '{}/result_{}_{}_{}.pkl'.format(result_folder, data_name, m, c))

    # emb = gwd_model.gwl_model.emb_model[0]

    # indxx = torch.LongTensor([0, 1])
    # # print(emb(indxx))
    # print(emb)
    print(gwd_model.obtain_embedding(
        opt_dict, torch.LongTensor([0, 1]), 0))
    # print(data_mc3['src_index'].keys())
    # print(data_mc3['src_index'].keys()[0])

    # print(gwd_model.gwl_model.emb_model)
    # print(gwd_model.gwl_model.emb_model[0])
    # # print(gwd_model.gwl_model.emb_model[1])
    # print(gwd_model.d_gw)
    # print(gwd_model.d_gw[0])
    # print(gwd_model.gwl_model.emb_model[0](gwd_model.d_gw[0]))
    # print(gwd_model.gwl_model.emb_model[1](0))
