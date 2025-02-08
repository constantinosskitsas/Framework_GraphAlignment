import numpy as np
from scipy.sparse.linalg import eigsh, eigs
from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse.linalg import LinearOperator
import torch
from scipy.sparse.csgraph import shortest_path

def solveDSpp(D1, D2, problemType):
    n = D1.shape[0]
    k = D2.shape[0]
    params = {
        'n': n,
        'k': k,
        'problemType': problemType,
        'D1': D1,
        'D2': D2
    }

    translationVec = np.linspace(0, 1.1, 10)
    params['injective'] = getoptions(params, 'injective', not (n == k))
    params['linTerm'] = getoptions(params, 'linTerm', colstack(np.zeros((k, n))))

    lambMin, lambMax = findLambdaMinAndMaxFast(params)  # Assuming findLambdaMinAndMaxFast exists
    maximalTranslation = -np.real(lambMin)
    minimalTranslation = -np.real(lambMax)
    X_opt_arr = []
    for ii in range(len(translationVec)):
        t = translationVec[ii]
        params['translation'] = t * minimalTranslation + (1 - t) * maximalTranslation
        X_opt, local_opt_obj = solveDsQuadprog(params)  # Assuming solveDsQuadprog exists
        if ii == 1:
            opt_obj = local_opt_obj
        X_opt_arr.append(X_opt)
        params['Xinit'] = X_opt

    objs = np.zeros(len(translationVec))
    for ii, X in enumerate(X_opt_arr):
        objs[ii] = X.flatten().dot(calcWProdFast(X.flatten(), 0, params))  # Assuming calcWProdFast exists

    final_obj = X_opt.flatten().dot(calcWProdFast(X_opt.flatten(), 0, params))

    if params['injective']:
        X_opt = X_opt[1:k + 1, :]
    maxInd = np.argmax(X_opt, axis=1)
    X_proj = np.zeros((k, n))
    X_proj[np.arange(k), maxInd] = 1

    return X_proj, X_opt, final_obj, opt_obj

def getoptions(options, name, default, mandatory=False):
    if name in options:
        return options[name]
    elif mandatory:
        raise ValueError(f"Option '{name}' is required.")
    else:
        return default

def calcWProdFast(v, Wtranslation, params):
    m = params['k']
    n = params['n']

    if params['problemType'] == 'L1':
        if params['injective']:
            Xaug = v.reshape(m + 1, n)
            outMat = np.zeros_like(Xaug)
            X = Xaug[1:, :]
            x = X.flatten()
            subVec = params['W'] @ x
            subMat = subVec.reshape(m, n)
            outMat[1:, :] = subMat
            out2 = outMat.flatten()
        else:
            out2 = params['W'] @ v
    elif params['problemType'] == 'GW':
        if params['injective']:
            Xaug = v.reshape(m + 1, n)
            outMat = np.zeros_like(Xaug)
            X = Xaug[1:, :]
            subMat = -2 * params['D2'] @ X @ params['D1'] + np.ones((m, 1)) * (np.ones((1, m)) @ X @ params['D1'] ** 2)
            outMat[1:, :] = subMat  # + Wtranslation * X
            out2 = outMat.flatten()
        else:
            out2 = -(colstack((params['D2'] @ v.reshape(n, n) @ params['D1']).T))
    else:
        raise ValueError("Invalid problemType")

    out = out2 + Wtranslation * v
    return out

def colstack(X):
    return X.flatten()

def findLambdaMinAndMaxFast(params):
    n = params['n']
    m = params['k']

    # Preprocessing
    if params['injective']:
        A_ds, _ = getDoublyStochasticConstraints(m + 1, n)  # Assuming getDoublyStochasticConstraints exists
        A = A_ds.T
        A = A[:, :n + m]
        vDim = (m + 1) * n
    else:
        A_ds, _ = getDoublyStochasticConstraints(n)
        A = A_ds.T
        A = A[:, :2 * n - 2]
        vDim = n**2

    Ginv = np.linalg.inv(A.T @ A)
    preprocessed = {'Ginv': Ginv, 'A': A}
    # print(A.shape)
    # Find lambaMax
    opt = {'tol': 1e-6}
    funHandle = lambda v: eigHelper(preprocessed, v, params)
    lambMaxValue, _ = eigs(LinearOperator((vDim,vDim), matvec=funHandle), k=1, which='LM', tol=opt['tol'])
    # _, lambMaxValue = eigs(eigHelper(preprocessed, vDim, params), k=1, which='LM', tol=opt['tol'])
    lambMaxValue = lambMaxValue[0]  # Extract eigenvalue from array

    if lambMaxValue > 0:
        lambMax = lambMaxValue
        opt['tol'] = 1e-6 * lambMax
        Shift = 1.1 * lambMax
        newFunHandle = lambda v: Shift * projectAffinedDS(v, preprocessed) - eigHelper(preprocessed, v, params)
        lambMinShifted, _ = eigs(LinearOperator((vDim,vDim), matvec=funHandle), k=1, which='LM', tol=opt['tol'])
        lambMin = Shift - lambMinShifted[0]
    else:
        lambMin = lambMaxValue
        opt['tol'] = 1e-4 * np.abs(lambMin)
        Shift = 1.1 * lambMin
        newFunHandle = lambda v: eigHelper(preprocessed, v, params) - Shift * projectAffinedDS(v, preprocessed)
        lambMaxShifted, _ = eigs(LinearOperator((vDim,vDim), matvec=funHandle), k=1, which='LM', tol=opt['tol'])
        lambMax = lambMaxShifted[0] + Shift

    return lambMin, lambMax

def eigHelper(preprocessed, v, params):
    u = projectAffinedDS(v, preprocessed)
    u = calcWProdFast(u, 0, params)
    u = projectAffinedDS(u, preprocessed)
    return u

def projectAffinedDS(v, preprocessed):
    Pv = preprocessed['A'].T @ v
    Pv = preprocessed['Ginv'] @ Pv
    Pv = preprocessed['A'] @ Pv
    u = v - Pv
    return u


def getDoublyStochasticConstraints(n, k=None):
    """
    Represents the affine doubly stochastic constraints Ax = b where x is the column-stacked DS matrix X.

    Args:
        n: Number of rows of the doubly stochastic matrix.
        k: Number of columns of the doubly stochastic matrix. Defaults to n.

    Returns:
        A: Constraint matrix.
        b: Constraint vector.
    """

    if k is None:
        k = n
    
    T = getTensorTranspose(n, k)
    CRows = np.kron(np.ones((k, 1)).T, np.eye(n))
    CCols = np.kron(np.ones((n, 1)).T, np.eye(k)) @ T
    A = np.vstack((CRows, CCols))
    b = np.ones(n + k)
    return A, b

# def getTensorTranspose(n, m):
#     """
#     Returns a matrix T such that vec(X') = T * vec(X) for any nxm matrix X.

#     Args:
#         n: Number of rows of the matrix.
#         m: Number of columns of the matrix.

#     Returns:
#         T: Tensor transpose matrix.
#     """

#     i, j = np.meshgrid(np.arange(n), np.arange(m))
#     # T = np.sparse(j.flatten(), i.flatten(), np.ones((n * m)), m * n, n * m)
#     T = csr_matrix(((j.flatten(), i.flatten()), np.ones((n * m))), shape=(m * n, n * m))
#     return T

def getTensorTranspose(n, m):
    """
    Returns a sparse matrix T such that vec(X.T) = T * vec(X) for any nxm matrix X.

    Args:
        n (int): Number of rows in the matrix.
        m (int): Number of columns in the matrix.

    Returns:
        scipy.sparse.coo_matrix: The sparse matrix T.
    """
    # Create meshgrid
    i, j = np.meshgrid(np.arange(n), np.arange(m), indexing='ij')
    
    # Compute row and column indices for the sparse matrix
    row_indices = np.ravel_multi_index((j, i), (m, n))
    col_indices = np.ravel_multi_index((i, j), (n, m))
    
    # Convert indices to 1D arrays
    row_indices = np.ravel(row_indices)
    col_indices = np.ravel(col_indices)
    
    # Create data array
    data = np.ones_like(row_indices, dtype=float)
    
    # Create sparse matrix
    T = coo_matrix((data, (row_indices, col_indices)), shape=(m * n, n * m))
    
    return T

def myEntropy(x):
    """
    Computes the entropy of the matrix x.

    Args:
        x: Input matrix.

    Returns:
        Entropy of the matrix.
    """

    if x.shape[0] < x.shape[1]:
        x = x.T
    # print(x)
    # assert np.all(x >= 0)
    zero_idx = x == 0
    return -np.sum(x[~zero_idx] * np.log(x[~zero_idx]))

def sinkhorn_custom(C, params, exp = False, reg=1, maxIter=10, stopThr=1e-9,
                   verbose=False, log=False, warm_start=None, eval_freq=10, print_freq=200, **kwargs):


    a = torch.ones((C.size()[0],), dtype = torch.float64) / C.size()[0]
    b = torch.ones((C.size()[1],), dtype = torch.float64) / C.size()[1]
    # else:
    #     a = torch.ones(C.size()[0], dtype = torch.float64)
    #     b = torch.ones(C.size()[1], dtype = torch.float64)

    na, nb = C.shape

    assert na >= 1 and nb >= 1, 'C needs to be 2d'
    assert na == a.shape[0] and nb == b.shape[0], "Shape of a or b does't match that of C"
    assert reg > 0, 'reg should be greater than 0'
    assert a.min() >= 0. and b.min() >= 0., 'Elements in a or b less than 0'

    if log:
        log = {'err': []}

    u = torch.from_numpy(getoptions(params, 'v', np.ones(na)))
    v = torch.from_numpy(getoptions(params, 'w', np.ones(nb)))

    # if warm_start is not None:
    #     u = warm_start['u']
    #     v = warm_start['v']
    # else:
    #     u = torch.ones(na, dtype=a.dtype) / na
    #     v = torch.ones(nb, dtype=b.dtype) / nb

    if exp:
        K = torch.empty(C.shape, dtype=C.dtype)
        K = -C/reg
        torch.exp(K, out=K)
    else:
        K = C

    b_hat = torch.empty(b.shape, dtype=C.dtype)

    it = 1
    err = 1

    # allocate memory beforehand
    KTu = torch.empty(v.shape, dtype=v.dtype)
    Kv = torch.empty(u.shape, dtype=u.dtype)

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
    M_EPS = 1e-16
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
        return P, u, v
    
def sinkhorn(K, params):
    maxSinkhornIter = getoptions(params, 'maxSinkhornIter', 100)
    verbose = getoptions(params, 'verbose', False)
    n = K.shape[0]
    sinkhornTol = getoptions(params, 'sinkhornTol', 1e-6)
    v = getoptions(params, 'v', np.ones(n))
    w = getoptions(params, 'w', np.ones(n))
    dist_v = np.full(maxSinkhornIter, np.inf)
    dist_w = np.full(maxSinkhornIter, np.inf)
    Kw = K @ w

    for i in range(maxSinkhornIter):
        if np.any(Kw == 0):
            raise ValueError('division by zero')
        v = 1 / Kw
        Kv = K.T @ v
        if np.any(Kv == 0):
            raise ValueError('division by zero')
        w = 1 / Kv
        Kw = K @ w

        # compute distance from correct marginals
        dist_v[i] = np.log(np.max(w * Kv) / np.min(w * Kv))
        dist_w[i] = np.log(np.max(v * Kw) / np.min(v * Kw))

        # Stop when margins are close to 1
        if i > 1:
            ratio = dist_w[i] / dist_w[i - 1]
            if ratio > 0.9:
                break

        if dist_v[i] < sinkhornTol and dist_w[i] < sinkhornTol:
            break

    Mv = np.tile(v, (n, 1))
    Mw = np.tile(w[:, np.newaxis], (1, n))
    X = Mv * K * Mw
    if verbose:
        print(f'Sinkhorn projection ended after {i} iterations...')
    iter_num = i
    return X, v, w#, iter_num, dist_v, dist_w

def solveDsQuadprog(params):
    n = getoptions(params, 'n', 6)
    k = getoptions(params, 'k', n)
    maxIter = getoptions(params, 'maxIter', 20)
    eta = getoptions(params, 'eta', 0.01)
    alpha = getoptions(params, 'alpha', 0.01)
    tol = getoptions(params, 'tol', 1e-8)

    if params['injective']:
        Xinit = np.ones((k + 1, n))
        # Xinit = sinkhornInjective(Xinit, params)  # Assuming sinkhornInjective exists
        Xinit = getoptions(params, 'Xinit', Xinit)
    else:
        Xinit = getoptions(params, 'Xinit', np.ones((n, n)) / n)

    linTerm = getoptions(params, 'linTerm', colstack(np.zeros((k, n))))
    if params['injective']:
        helperMat = np.zeros((k + 1, n))
        helperMat[1:, :] = linTerm.reshape(k, n)
        linTerm = helperMat.flatten()
    Wtranslation = params['translation']

    running = True
    ii = 0
    oldX = Xinit
    objs = np.full(maxIter, np.inf)

    while running:
        ii += 1
        expArg = -(calcWProdFast(oldX.flatten(), Wtranslation, params) + linTerm)
        originalExpArg = expArg
        expArg -= np.max(expArg)
        scaleFactor = np.max(np.abs(expArg))
        alphaNormalized = alpha * scaleFactor
        
        fX = np.reshape(np.exp(expArg / alphaNormalized), oldX.shape)
        K = (fX ** eta) * (oldX ** (1 - eta))

        if params['injective']:
            print("injective")
            # currX, v, w = sinkhornInjective(K, params)  # Assuming sinkhornInjective exists
        else:
            currX, v, w = sinkhorn_custom(torch.from_numpy(K), params)
            currX = currX.numpy()
            v = v.numpy()
            w = w.numpy()
            # currX, v, w = sinkhorn(K, params)

        params['v'] = v
        params['w'] = w

        # print(currX.shape)
        objs[ii] = -originalExpArg.dot(currX.flatten()) - alphaNormalized * myEntropy((currX.flatten()).reshape(-1, 1))
        # assert (not np.isnan(objs[ii])) and (not np.isinf(objs[ii]))

        oldX = currX

        if ii > 1 and not np.isnan(objs[ii]) and not np.isnan(objs[ii - 1]):
            change = np.abs(objs[ii] - objs[ii - 1]) / np.abs(objs[ii])
        else:
            change = np.inf

        running = (ii < (maxIter - 1)) and (change > tol)
        # print(ii, running)

    X = currX
    obj_opt = X.flatten().dot(calcWProdFast(X.flatten(), Wtranslation, params)) - Wtranslation * k
    return X, obj_opt

def main(data):
    print("Dspp")
    A = data['Src']
    B = data['Tar']
    D1 = shortest_path(A)
    D2 = shortest_path(B)
    X_proj, X_opt, final_obj, opt_obj = solveDSpp(D1, D2, 'GW')
    return X_opt,X_opt