from sacred import Experiment
from sacred.observers import FileStorageObserver
import logging
from algorithms import gwl, conealign, grasp as grasp, regal, eigenalign, NSD, isorank2 as isorank, netalign, klaus, sgwl,Grampa,GraspB,GrampaS

ex = Experiment("ex")

ex.observers.append(FileStorageObserver('runs'))

# create logger
logger = logging.getLogger('e')
logger.setLevel(logging.INFO)
logger.propagate = False

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# create formatter
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)

ex.logger = logger


_GW_args = {
    'opt_dict': {
        'epochs': 1,
        'batch_size': 1000000,
        'use_cuda': False,
        'strategy': 'soft',
        # 'strategy': 'hard',
        # 'beta': 0.1,
        'beta': 1e-1,
        'outer_iteration': 400,  # M
        'inner_iteration': 1,  # N
        'sgd_iteration': 300,
        'prior': False,
        'prefix': 'results',
        'display': False
    },
    'hyperpara_dict': {
        'dimension': 90,
        # 'loss_type': 'MSE',
        'loss_type': 'L2',
        'cost_type': 'cosine',
        # 'cost_type': 'RBF',
        'ot_method': 'proximal'
    },
    # 'lr': 0.001,
    'lr': 1e-3,
    # 'gamma': 0.01,
    # 'gamma': None,
    'gamma': 0.8,
    # 'max_cpu': 20,
    # 'max_cpu': 4
}

_SGW_args = {
    'ot_dict': {
        'loss_type': 'L2',  # the key hyperparameters of GW distance
        'ot_method': 'proximal',
         #'beta': 0.025,#euroroad
         #'beta': 0.025,#netscience,eurorad,arenas
        #'beta': 0.1,#dense ex fb, socfb datasets
        'beta': 0.025,# 0.025-0.1 depends on degree
        # outer, inner iteration, error bound of optimal transport
        'outer_iteration': 2000,  # num od nodes
        'iter_bound': 1e-10,
        'inner_iteration': 2,
        'sk_bound': 1e-30,
        'node_prior': 1000,
        'max_iter': 4,  # iteration and error bound for calcuating barycenter
        'cost_bound': 1e-26,
        'update_p': False,  # optional updates of source distribution
        'lr': 0,
        'alpha': 0
    },
    "mn": 1,  # gwl
    # "mn": 1,  # s-gwl-3
    # "mn": 2,  # s-gwl-2
    # "mn": 3,  # s-gwl-1
    'max_cpu': 20,
}

_CONE_args = {
    'dim': 512,  # clipped by Src[0] - 1
    'window': 10,
    'negative': 1.0,
    'niter_init': 10,
    'reg_init': 1.0,
    'nepoch': 5,
    'niter_align': 10,
    'reg_align': 0.05,
    'bsz': 10,
    'lr': 1.0,
    'embsim': 'euclidean',
    'alignmethod': 'greedy',
    'numtop': 10
}

_GRASP_args = {
    'laa': 2,
    'icp': False,
    'icp_its': 3,
    'q': 100,
    'k': 20,
    #'n_eig': Src.shape[0] - 1
    'n_eig': 100,
    'lower_t': 1.0,
    'upper_t': 50.0,
    'linsteps': True,
    'base_align': True
}

_REGAL_args = {
    'attributes': None,
    'attrvals': 2,
    'dimensions': 128,  # useless
    'k': 10,            # d = klogn
    'untillayer': 2,    # k
    'alpha': 0.01,      # delta
    'gammastruc': 1.0,
    'gammaattr': 1.0,
    'numtop': 10,
    'buckets': 2
}

_LREA_args = {
    'iters': 40,
    'method': "lowrank_svd_union",
    'bmatch': 3,
    'default_params': True
}

_NSD_args = {
    'alpha': 0.8,
    'iters': 20
}

_ISO_args = {
    'alpha': 0.9,
    'tol': 1e-12,
    'maxiter': 100,
    'lalpha': 10000,
    'weighted': True
}

_NET_args = {
    'a': 1,
    'b': 2,
    'gamma': 0.95,
    'dtype': 2,
    'maxiter': 100,
    'verbose': True
}

_KLAU_args = {
    'a': 1,
    'b': 1,
    'gamma': 0.4,
    'stepm': 25,
    'rtype': 2,
    'maxiter': 100,
    'verbose': True
}
_Grampa_args = {
   # 'eta': 0.2
   'eta': 0.2,
   'lalpha':10000,
   'initSim':0,
   'Eigtype':0
}
_GrampaS_args = {
    'eta': 0.2,
    'lalpha':10000
}
_GRASPB_args = {
    'laa': 3,
    'icp': True,
    'icp_its': 3,
    'q': 20,
    'k': 20,
    #'n_eig': Src.shape[0] - 1
    #n_eig': 100,
    'lower_t': 0.1,
    'upper_t': 50.0,
    'linsteps': True,
    'ba_': True,
    'corr_func': 3,
    'k_span':40

}


_algs = [
    (gwl, _GW_args, [3], "GW"),
    (conealign, _CONE_args, [-3], "CONE"),
    (grasp, _GRASP_args, [-3], "GRASP"),
    (regal, _REGAL_args, [-3], "REGAL"),
    (eigenalign, _LREA_args, [3], "LREA"),
    (NSD, _NSD_args, [30], "NSD"),

    (isorank, _ISO_args, [3], "ISO"),
    (netalign, _NET_args, [3], "NET"),
    (klaus, _KLAU_args, [3], "KLAU"),
    (sgwl, _SGW_args, [3], "SGW"),
    (Grampa, _Grampa_args, [3], "GRAMPA"),
    (GraspB, _GRASPB_args, [-96], "GRASPB"),
    (GrampaS, _GrampaS_args, [4], "GRAMPAS"),
]   

_acc_names = [
    "acc",
    "EC",
    "ICS",
    "S3",
    "jacc",
    "mnc",
]
