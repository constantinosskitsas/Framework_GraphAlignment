import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.linalg import kron
from scipy.sparse import kron, csr_matrix

# Function to calculate Frobenius norm
def frobenius_norm(A):
    return np.linalg.norm(A, 'fro')

# Function to calculate F0
def F0(P, A_G, A_H):
    return frobenius_norm(A_G @ P - P @ A_H)**2

# Function to calculate F1
def F1(P, Delta, L_H, L_G):
    return -np.trace(Delta @ P) - 2 * (P.flatten() @ kron(L_H.T, L_G.T) @ P.flatten())

# Function to calculate F_lambda
def F_lambda(P, A_G, A_H, Delta, L_H, L_G, lambda_):
    return F0(P, A_G, A_H) + lambda_ * F1(P, Delta, L_H, L_G)

def kron_mv(L_H, L_G, v):
    """Compute (kron(L_H.T, L_G.T) @ v) without forming the full Kronecker matrix."""
    n, m = L_H.shape
    p, q = L_G.shape
    V = v.reshape(q, m)
    result = L_G.T @ V @ L_H
    return result.ravel()
# Frank-Wolfe algorithm to find P* for minimizing F0
def frank_wolfe(A_G, A_H, Delta, L_H, L_G, lambda_, max_iter=10, tol=1e-6):
    n = A_G.shape[0]
    P = np.ones((n, n)) / n  # Start with a doubly stochastic matrix
    for _ in range(max_iter):
        # Gradient of F0
        grad_F0 = 2 * (A_G.T @ (A_G @ P - P @ A_H) - (A_G @ P - P @ A_H) @ A_H.T)
        
        # Gradient of F1
        grad_F1 = -Delta - 4 * (kron(L_H.T, L_G.T) @ P.flatten()).reshape(n, n)
        
        # Combined gradient
        grad = grad_F0 + lambda_ * grad_F1

        row_ind, col_ind = linear_sum_assignment(grad)
        S = np.zeros_like(P)
        S[row_ind, col_ind] = 1
        # S = sinkhorn(ones, ones, G, reg, maxIter = 500, stopThr = 1e-3)
        gamma = 2 / (2 + _)
        P_new = (1 - gamma) * P + gamma * S
        if frobenius_norm(P_new - P) < tol:
            break
        P = P_new
    return P
def frank_wolfe_sparse(A_G, A_H, Delta, L_H, L_G, lambda_, max_iter=10, tol=1e-6):
    n = A_G.shape[0]
    P = np.ones((n, n)) / n  # Start with a doubly stochastic matrix
    for _ in range(max_iter):
        # Gradient of F0
        grad_F0 = 2 * (A_G.T @ (A_G @ P - P @ A_H) - (A_G @ P - P @ A_H) @ A_H.T)
        
        # Gradient of F1
        grad_F1 = -Delta - 4 * (kron(L_H.T, L_G.T, format='csr') @ P.flatten()).reshape(n, n)
        
        # Combined gradient
        grad = grad_F0 + lambda_ * grad_F1

        row_ind, col_ind = linear_sum_assignment(grad)
        S = np.zeros_like(P)
        S[row_ind, col_ind] = 1
        # S = sinkhorn(ones, ones, G, reg, maxIter = 500, stopThr = 1e-3)
        gamma = 2 / (2 + _)
        P_new = (1 - gamma) * P + gamma * S
        if frobenius_norm(P_new - P) < tol:
            break
        P = P_new
    return P
# Main PATH algorithm
def path_algorithm(A_G, A_H, Delta, L_H, L_G, d_lambda=0.1, epsilon=1e-3):
    # Step 1: Initialization
    lambda_ = 0
    L_H_sparse = csr_matrix(L_H)
    L_G_sparse = csr_matrix(L_G)
    A_G_sparse = csr_matrix(A_G)
    A_H_sparse = csr_matrix(A_H)
    #P = frank_wolfe(A_G, A_H, Delta, L_H, L_G, lambda_)
    P = frank_wolfe_sparse(A_G, A_H, Delta, L_H_sparse, L_G_sparse, lambda_)
    # Step 2: Cycle over lambda
    while lambda_ < 1:
        lambda_new = lambda_ + d_lambda
        if lambda_new > 1:
            lambda_new = 1
        
        F_old = F_lambda(P, A_G, A_H, Delta, L_H, L_G, lambda_)
        F_new = F_lambda(P, A_G, A_H, Delta, L_H, L_G, lambda_new)
        
        if np.abs(F_new - F_old) < epsilon:
            lambda_ = lambda_new
        else:
            P = frank_wolfe_sparse(A_G, A_H, Delta, L_H, L_G, lambda_new)
            lambda_ = lambda_new

    # Step 3: Output
    P_out = P
    return P_out

def laplacian_matrix(adj_matrix):
    degree_matrix = np.diag(np.sum(adj_matrix, axis=1))
    return degree_matrix - adj_matrix

def main(data):
    print("path")
    A_H = data['Src']
    A_G = data['Tar']
    D_H = np.diag(np.sum(A_H, axis=1))
    D_G = np.diag(np.sum(A_G, axis=1))
    Delta = (D_H - D_G)**2
    L_H = laplacian_matrix(A_H)
    L_G = laplacian_matrix(A_G)

    P_out = path_algorithm(A_G, A_H, Delta, L_H, L_G)
    return P_out,P_out