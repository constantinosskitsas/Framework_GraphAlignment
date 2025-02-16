import numpy as np
from scipy import sparse




#Full NMF matrix (which NMF factorizes with SVD)
#Taken from MILE code
def netmf_mat_full(A, window = 10, b=1.0):
    if not sparse.issparse(A):
        A = sparse.csr_matrix(A)
    #print "A shape", A.shape
    n = A.shape[0]
    vol = float(A.sum())
    L, d_rt = sparse.csgraph.laplacian(A, normed=True, return_diag=True)
    X = sparse.identity(n) - L
    S = np.zeros_like(X)
    X_power = sparse.identity(n)
    for i in range(window):
        #print "Compute matrix %d-th power" % (i + 1)
        X_power = X_power.dot(X)
        S += X_power
    S *= vol / window / b
    D_rt_inv = sparse.diags(d_rt ** -1)
    M = D_rt_inv.dot(D_rt_inv.dot(S).T)
    result = np.log(np.maximum(M.todense(),1))
    return sparse.csr_matrix(result)

#Used in NetMF, AROPE
def svd_embed(prox_sim, dim):

    # 无语，sparse。linalg.svds居然有误差，同一个矩阵分解出来的差很多，SOS
    u, s, v = sparse.linalg.svds(prox_sim, dim, return_singular_vectors="u")

    # u,s,v = np.linalg.svd(prox_sim.todense(),dim)
    return sparse.diags(np.sqrt(s)).dot(u.T).T



def netmf(A, dim = 128, window=5, b=1.0, normalize = True):
    prox_sim = netmf_mat_full(A, window, b)
    embed = svd_embed(prox_sim, dim)
    if normalize:
        norms = np.linalg.norm(embed, axis = 1).reshape((embed.shape[0], 1))
        norms[norms == 0] = 1
        embed = embed / norms
    return embed