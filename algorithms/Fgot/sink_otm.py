import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import pandas as pd

def sink_vect_exp(K, numItermax=10, stopThr=1e-9, cuda = False):

    # we assume that no distances are null except those of the diagonal of
    # distances

    samples = K.size()[0]
    a = torch.ones((K.size()[1],)) / K.size()[1]
    b = torch.ones(( K.size()[2],)) / K.size()[2]
    b_scalar = b[0]
    
    # init data
    Nini = len(a)
    Nfin = len(b)

    u = torch.ones(samples, Nini, 1) / Nini
    v = torch.ones(samples, Nfin, 1) / Nfin


    Kp = (1 / a).view(-1, 1) * K

    cpt = 0
    err = 1
    while (err > stopThr and cpt < numItermax): 
        uprev = u
        vprev = v
        KtransposeU = torch.matmul(K.transpose(1,2), u)
        v = b_scalar/KtransposeU 
        u = 1. / (Kp @ v)
        
        if cpt % 10 == 0:
            transp = u * (K * v.transpose(1,2))
            err = (torch.sum(transp) - b).norm(1).pow(2).item()

        cpt += 1
    return u * K * v.transpose(1,2)



def sink_vect(M, reg, numItermax=10, stopThr=1e-9, cuda = False):

    # we assume that no distances are null except those of the diagonal of
    # distances


    samples = M.size()[0]
    a = torch.ones((M.size()[1],)) / M.size()[1]
    b = torch.ones(( M.size()[2],)) / M.size()[2]
    b_scalar = b[0]
    
    # init data
    Nini = len(a)
    Nfin = len(b)

    u = torch.ones(samples, Nini, 1) / Nini
    v = torch.ones(samples, Nfin, 1) / Nfin


    K = torch.exp(-M / reg)

    Kp = (1 / a).view(-1, 1) * K

    cpt = 0
    err = 1
    while (err > stopThr and cpt < numItermax): 
        uprev = u
        vprev = v
        KtransposeU = torch.matmul(K.transpose(1,2), u) 
        v = b_scalar/KtransposeU
        u = 1. / (Kp @ v)
        
        if cpt % 10 == 0:
            transp = u * (K * v.transpose(1,2))
            err = (torch.sum(transp) - b).norm(1).pow(2).item() / samples

        cpt += 1
    return u * K * v.transpose(1,2)



def sink_exp(K, numItermax=10, stopThr=1e-9, cuda = False):

    # we assume that no distances are null except those of the diagonal of
    # distances

    if cuda:
        a = Variable(torch.ones((K.size()[0],)) / K.size()[0]).cuda()
        b = Variable(torch.ones((K.size()[1],)) / K.size()[1]).cuda()
    else:
        a = torch.ones((K.size()[0],)) / K.size()[0]
        b = torch.ones((K.size()[1],)) / K.size()[1]

    # init data
    Nini = len(a)
    Nfin = len(b)

    if cuda:
        u = Variable(torch.ones(Nini) / Nini).cuda()
        v = Variable(torch.ones(Nfin) / Nfin).cuda()
    else:
        u = torch.ones(Nini) / Nini
        v = torch.ones(Nfin) / Nfin


    Kp = (1 / a).view(-1, 1) * K
    cpt = 0
    err = 1
    while (err > stopThr and cpt < numItermax): 
        uprev = u
        vprev = v
        KtransposeU = K.t() @ u
        v = b / KtransposeU
        u = 1. / (Kp @ v)
        
        if torch.any(KtransposeU == 0) or torch.any(torch.isnan(u)) or torch.any(torch.isnan(v)) or \
                torch.any(torch.isinf(u)) or torch.any(torch.isinf(v)):
            print('Warning: numerical errors at iteration', cpt)
            u, v = uprev, vprev
            break

        if cpt % 10 == 0:
            transp = u.view(-1, 1) * (K * v)
            err = (torch.sum(transp) - b).norm(1).pow(2).item()

        cpt += 1
    return u.view((-1, 1)) * K * v.view((1, -1)) 

def sinkhorn_custom(C, exp, reg=1, maxIter=10, stopThr=1e-9,
                   verbose=False, log=False, warm_start=None, eval_freq=10, print_freq=200, **kwargs):


    a = torch.ones((C.size()[0],)) / C.size()[0]
    b = torch.ones((C.size()[1],)) / C.size()[1]
    # else:
    #     a = torch.ones(C.size()[0], dtype = torch.float64)
    #     b = torch.ones(C.size()[1], dtype = torch.float64)

    device = a.device
    na, nb = C.shape

    assert na >= 1 and nb >= 1, 'C needs to be 2d'
    assert na == a.shape[0] and nb == b.shape[0], "Shape of a or b does't match that of C"
    assert reg > 0, 'reg should be greater than 0'
    assert a.min() >= 0. and b.min() >= 0., 'Elements in a or b less than 0'

    if log:
        log = {'err': []}

    if warm_start is not None:
        u = warm_start['u']
        v = warm_start['v']
    else:
        u = torch.ones(na, dtype=a.dtype).to(device) / na
        v = torch.ones(nb, dtype=b.dtype).to(device) / nb

    if exp:
        K = torch.empty(C.shape, dtype=C.dtype).to(device)
        K = -C/reg
        torch.exp(K, out=K)
    else:
        K = C

    b_hat = torch.empty(b.shape, dtype=C.dtype).to(device)

    it = 1
    err = 1

    # allocate memory beforehand
    KTu = torch.empty(v.shape, dtype=v.dtype).to(device)
    Kv = torch.empty(u.shape, dtype=u.dtype).to(device)

    # t1 = time.time()
    while (err > stopThr and it <= maxIter):
        upre, vpre = u, v
        KTu = torch.matmul(u, K)
        v = torch.div(b, KTu)
        Kv = torch.matmul(K, v)
        u = torch.div(a, Kv)

        if torch.any(KTu == 0) or torch.any(torch.isnan(u)) or torch.any(torch.isnan(v)) or \
                torch.any(torch.isinf(u)) or torch.any(torch.isinf(v)):
            print('Warning: numerical errors at iteration', it)
            u, v = upre, vpre
            break

        it += 1
    # t2 = time.time()
    # print("Sinkhorn loop: ", t2 - t1)

    if log:
        log['u'] = u
        log['v'] = v
        log['alpha'] = reg * torch.log(u + M_EPS)
        log['beta'] = reg * torch.log(v + M_EPS)

    # transport plan
    P = u.reshape(-1, 1) * K * v.reshape(1, -1)
    if log:
        return P, log
    else:
        return P

def sink(M, reg, numItermax=10, stopThr=1e-9, cuda = False):

    # we assume that no distances are null except those of the diagonal of
    # distances

    if cuda:
        a = Variable(torch.ones((M.size()[0],)) / M.size()[0]).cuda()
        b = Variable(torch.ones((M.size()[1],)) / M.size()[1]).cuda()
    else:
        a = torch.ones((M.size()[0],)) / M.size()[0]
        b = torch.ones((M.size()[1],)) / M.size()[1]

    # init data
    Nini = len(a)
    Nfin = len(b)

    if cuda:
        u = Variable(torch.ones(Nini) / Nini).cuda()
        v = Variable(torch.ones(Nfin) / Nfin).cuda()
    else:
        u = torch.ones(Nini) / Nini
        v = torch.ones(Nfin) / Nfin

    K = torch.exp(-M / reg)

    Kp = (1 / a).view(-1, 1) * K
    cpt = 0
    err = 1
    while (err > stopThr and cpt < numItermax): 
        uprev = u
        vprev = v
        KtransposeU = K.t() @ u
        v = b / KtransposeU
        u = 1. / (Kp @ v)
        
        if torch.any(KtransposeU == 0) or torch.any(torch.isnan(u)) or torch.any(torch.isnan(v)) or \
                torch.any(torch.isinf(u)) or torch.any(torch.isinf(v)):
            print('Warning: numerical errors at iteration', cpt)
            u, v = uprev, vprev
            break

        if cpt % 10 == 0:
            transp = u.view(-1, 1) * (K * v)
            err = (torch.sum(transp) - b).norm(1).pow(2).item()

        cpt += 1
    return u.view((-1, 1)) * K * v.view((1, -1)) 


def sink_stabilized(M, reg, numItermax=1000, tau=1e2, stopThr=1e-9, warmstart=None, print_period=20, cuda=False):

    if cuda:
        a = Variable(torch.ones((M.size()[0],)) / M.size()[0]).cuda()
        b = Variable(torch.ones((M.size()[1],)) / M.size()[1]).cuda()
    else:
        a = Variable(torch.ones((M.size()[0],)) / M.size()[0])
        b = Variable(torch.ones((M.size()[1],)) / M.size()[1])

    # init data
    na = len(a)
    nb = len(b)

    cpt = 0
    # we assume that no distances are null except those of the diagonal of
    # distances
    if warmstart is None:
        if cuda:
            alpha, beta = Variable(torch.zeros(na)).cuda(), Variable(torch.zeros(nb)).cuda()
        else:
            alpha, beta = Variable(torch.zeros(na)), Variable(torch.zeros(nb))
    else:
        alpha, beta = warmstart

    if cuda:
        u, v = Variable(torch.ones(na) / na).cuda(), Variable(torch.ones(nb) / nb).cuda()
    else:
        u, v = Variable(torch.ones(na) / na), Variable(torch.ones(nb) / nb)

    def get_K(alpha, beta):
        return torch.exp(-(M - alpha.view((na, 1)) - beta.view((1, nb))) / reg)

    def get_Gamma(alpha, beta, u, v):
        return torch.exp(-(M - alpha.view((na, 1)) - beta.view((1, nb))) / reg + torch.log(u.view((na, 1))) + torch.log(v.view((1, nb))))

    # print(np.min(K))

    K = get_K(alpha, beta)
    transp = K
    loop = 1
    cpt = 0
    err = 1
    while loop:

        uprev = u
        vprev = v

        # sinkhorn update
        v = torch.div(b, (K.t().matmul(u) + 1e-16))
        u = torch.div(a, (K.matmul(v) + 1e-16))

        # remove numerical problems and store them in K
        if torch.max(torch.abs(u)).data[0] > tau or torch.max(torch.abs(v)).data[0] > tau:
            alpha, beta = alpha + reg * torch.log(u), beta + reg * torch.log(v)

            if cuda:
                u, v = Variable(torch.ones(na) / na).cuda(), Variable(torch.ones(nb) / nb).cuda()
            else:
                u, v = Variable(torch.ones(na) / na), Variable(torch.ones(nb) / nb)

            K = get_K(alpha, beta)

        if cpt % print_period == 0:
            transp = get_Gamma(alpha, beta, u, v)
            err = (torch.sum(transp) - b).norm(1).pow(2).data[0]

        if err <= stopThr:
            loop = False

        if cpt >= numItermax:
            loop = False

        #if np.any(np.isnan(u)) or np.any(np.isnan(v)):
        #    # we have reached the machine precision
        #    # come back to previous solution and quit loop
        #    print('Warning: numerical errors at iteration', cpt)
        #    u = uprev
        #    v = vprev
        #    break

        cpt += 1

    return torch.sum(get_Gamma(alpha, beta, u, v)*M)

def pairwise_distances(x, y, method='l1'):
    n = x.size()[0]
    m = y.size()[0]
    d = x.size()[1]

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    if method == 'l1':
        dist = torch.abs(x - y).sum(2)
    else:
        dist = torch.pow(x - y, 2).sum(2)

    return dist.float()

def dmat(x,y):
    mmp1 = torch.stack([x] * x.size()[0])
    mmp2 = torch.stack([y] * y.size()[0]).transpose(0, 1)
    mm = torch.sum((mmp1 - mmp2) ** 2, 2).squeeze()

    return mm