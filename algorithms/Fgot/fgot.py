
from algorithms.Fgot.sink_otm import *

from algorithms.Fgot.bregman import *

import numpy as np
import torch
torch.set_default_tensor_type('torch.DoubleTensor')
import random
import math
import numpy.linalg as lg
import scipy.linalg as slg
from matplotlib import pyplot as plt
from   numpy import linalg as LA

import networkx as nx
import time
def regularise_invert_one(x, alpha, ones):
    if ones:
        x_reg = lg.inv(x   + alpha * np.eye(len(x)) + np.ones([len(x),len(x)])/len(x)) 
    else:
        x_reg = lg.pinv(x) + alpha * np.eye(len(x))
    return x_reg
def get_filters(L1, method, tau = 0.2):
    if method == 'got':
        g1 = np.real(slg.sqrtm(regularise_invert_one(L1,alpha = 0.1, ones = False )))
    elif method == 'weight':
        g1 = np.diag(np.diag(L1)) - L1
    elif method == 'heat':
        g1 = slg.expm(-tau*L1)
    elif method == 'sqrtL':
        g1 = np.real(slg.sqrtm(L1))
    elif method == 'L':
        g1 = L1
    elif method == 'sq':
        g1 = L1 @ L1
    return g1

def loss(DS, g1, g2, loss_type, epsilon = 5e-4):
    """
    Calculate loss, with the help of initially calculated params
    """
     
    if loss_type == 'w_simple':
        cost = - 2 * torch.trace( g1 @ DS @ g2 @ DS.t() )
        
    elif loss_type == 'l2':       
        cost = torch.sum((g1 @ DS - DS @ g2)**2, dim=1).sum()
        
    return cost



# Algorithm -- Stochastic Mirror gradient 
#===================================================================
# lr = 1 is good
def main(data, tau=1, n_samples=5, epochs=1000, lr=1, 
            std_init = 5, loss_type = 'w_simple', seed=42, verbose=True, tol = 1e-12, adapt_lr = False):   
    print("fgot")
    g1 = data['Src']
    g2 = data['Tar']
    # Initialization
    torch.manual_seed(seed)
    g1 = get_filters(g1, 'sq')
    g2 = get_filters(g2, 'sq')
    n = g1.shape[0]
    m = g2.shape[0]
    # lr = 50*n*m
    if adapt_lr:
        lr = lr/(np.max(g1)*np.max(g2))
    g1 = to_torch(g1)
    g2 = to_torch(g2)
    
    # g1 = g1 - 0.5*torch.diag(torch.diag(g1))
    # g2 = g2 - 0.5*torch.diag(torch.diag(g2))
    
    
    mean = to_torch(np.outer(np.repeat(1/n, n), np.repeat(1/m, m)))
    mean = mean.requires_grad_()
    
    std  = std_init * torch.ones(n, m) 
    std  = std.requires_grad_() 
    
    history = []
    epoch = 0
    err = 1
    while (epoch < epochs): 
        cost = 0
        for sample in range(n_samples):
            eps = torch.rand(n, m) 
            P_noisy = mean + std * eps 
            proj = sinkhorn_custom(torch.relu(P_noisy) + 1/n, False)
            cost = cost + loss(proj, g1, g2, loss_type)
        cost = cost/n_samples
        cost.backward()
        
        # Aux.
        s2 = std.data**2
        d  = lr/2 * s2 * std.grad
        
        # Update
        mean_prev = mean.data
        mean.data = mean.data - lr * mean.grad * s2
        std.data  = torch.sqrt(s2 + d) - d   
        
        mean.grad.zero_()
        std.grad.zero_()
        
        # Tracking
        #history.append(cost.item())
        if ((epoch+1) % 10 == 0 and (epoch>50)):

            err = np.linalg.norm(sink(-tau*mean.detach(), tau) - sink(-tau*mean_prev.detach(), tau)) / (n*m)
        epoch = epoch + 1
    
    # return mean.detach()
    P = sinkhorn_custom(-tau*mean.detach(), True, reg=tau)
    
    # P = P.squeeze()
    # P = P.numpy()
    # P = mean
    
    
    # Convergence plot
#     if verbose:
#         plt.plot(history)
#         plt.show()
    return P,P


# Tools # ===================================================================================================================
   
    
def torch_invert(x,alpha, ones=False):
    if ones:
        return torch.inverse(x   + 
                             alpha * torch.from_numpy(np.eye(len(x))) + 
                             torch.from_numpy(np.ones([len(x),len(x)])/len(x)))
    else:
        return torch.inverse(x   + 
                             alpha * torch.from_numpy(np.eye(len(x))))

    
def to_torch(x):
    return torch.from_numpy(x.astype(np.float64))



def doubly_stochastic(P, tau, it):
    """ Uses logsumexp for numerical stability. """    
    A = P / tau
    for i in range(it):
        A = A - A.logsumexp(dim=1, keepdim=True)
        A = A - A.logsumexp(dim=0, keepdim=True) 
    return torch.exp(A)


def rnorm(M):
    r  = M.shape[0] 
    c  = M.shape[1]
    N  = np.zeros((r, c))

    
    for i in range(0,r):
        Mi=np.linalg.norm(M[i,:])
        if Mi!=0:
            N[i,:] = M[i,:] / Mi
        else:
            N[i,:] = M[i,:]
    return N  

