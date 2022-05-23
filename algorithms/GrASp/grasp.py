import numpy as np
import scipy as sci
import scipy.sparse as sps
import scipy.sparse.linalg as spsl
import networkx as nx
import time
import os
from sklearn.preprocessing import normalize
import argparse
import matplotlib.pyplot as plt
#import base_align as ba
#import munkres
from . import base_align as ba
#import base_align as ba

from sklearn.neighbors import NearestNeighbors
# np.set_printoptions(precision=4)

# folder="C:/Users/Judith/Documents/MATLAB/compare_matlab_python/"


# 	#noise_level= [1]
# 	scores=np.zeros([5,reps])
# 	for i in noise_level:
# 		for j in range(1,reps+1):
#
#
# 			print('Noise level %d, round %d' %(i,j))


# def parse_args():
#     parser = argparse.ArgumentParser(description="RUN GASP")
#     # parser.add_argument('--graph', nargs='?', default='Sacch')
#     # 1:nn 2:sortgreedy 3: jv
#     # parser.add_argument('--laa', type=int, default=2,
#     #                     help='Linear assignment algorithm. 1=nn,2=sortgreedy, 3=jv')
#     parser.add_argument('--icp', type=bool, default=False)
#     parser.add_argument('--icp_its', type=int, default=3,
#                         help='how many iterations of iterative closest point')
#     parser.add_argument('--q', type=int, default=100)
#     parser.add_argument('--k', type=int, default=20)
#     parser.add_argument('--n_eig', type=int, default=1132,
#                         help="how many eigenvectors to compute")
#     parser.add_argument('--lower_t', type=float, default=1.0,
#                         help='smallest timestep for corresponding functions')
#     parser.add_argument('--upper_t', type=float, default=50.0,
#                         help='biggest timestep for corresponding functions')
#     parser.add_argument('--linsteps', type=bool, default=True,
#                         help='scaling of time steps of corresponding functions, logarithmically or linearly')
#     # parser.add_argument('--reps', type=int, default=5,
#     #                     help='number of repetitions per noise level')
#     # parser.add_argument('--noise_levels', type=list, default=[10])
#     # parser.add_argument('--base_align', type=bool, default=True)
#     return parser.parse_known_args()[0]


def main(data, **args):  # alg=2, base_align=True):

    Src = data['Src']
    Tar = data['Tar']

    if args['n_eig'] is None:
        args['n_eig'] = Src.shape[0] - 1
        # args['n_eig'] = Src.shape[0]//100

    # args = parse_args()

    # # edge list first graph
    # edge_list_G1 = 'arenas_orig.txt'
    # # edge list second graph
    # edge_list_G2 = 'arenas_permutated.txt'

    # # ground truth file
    # gt_file = 'arenas_gt.txt'
    # #
    # data = np.loadtxt(gt_file, delimiter=" ")

    # data = data.astype(int)

    # gt = dict(data)

    # A1 = edgelist_to_adjmatrix(edge_list_G1)

    # A2 = edgelist_to_adjmatrix(edge_list_G2)

    # func_maps = functional_maps_base_align if base_align else functional_maps
    G1_emb, G2_emb = functional_maps_gen(Src, Tar, **args)
    # matching = func_maps(A1, A2, args.q, args.k, args.n_eig, args.laa, args.icp,
    #                      args.icp_its, args.lower_t, args.upper_t, args.linsteps)

    # if(args.base_align):
    #     matching = functional_maps_base_align(A1, A2, args.q, args.k, args.n_eig, alg,
    #                                           args.icp, args.icp_its, args.lower_t, args.upper_t, args.linsteps, args.graph)
    # else:
    #     matching = functional_maps(A1, A2, args.q, args.k, args.n_eig, alg, args.icp,
    #                                args.icp_its, args.lower_t, args.upper_t, args.linsteps, args.graph)

    # acc = eval_matching(matching, gt)
    # print('accuracy: %f' % acc)
    # print('\n')
    # return np.array(list(matching.items()), dtype=int).T
    return None, sci.spatial.distance_matrix(G1_emb.T, G2_emb.T)


def functional_maps_gen(A1, A2, q, k, n_eig, laa, icp, icp_its, lower_t, upper_t, linsteps, base_align):

    # 1:nn 2:sortgreedy 3: jv
    match = laa

    t = np.linspace(lower_t, upper_t, q)
    if (not linsteps):
        t = np.logspace(lower_t, upper_t, q)

    # n = np.shape(A1)[0]
    n = A1.shape[0]

    # decompose graph laplacians
    D1, V1 = decompose_laplacian(A1, True, n_eig)
    D2, V2 = decompose_laplacian(A2, True, n_eig)

    # calculate corresponding functions
    Cor1 = calc_corresponding_functions(n, q, t, D1, V1)
    Cor2 = calc_corresponding_functions(n, q, t, D2, V2)


    # calculate base alignment matrix
    if base_align:
        B = ba.optimize_AB(Cor1, Cor2, n, V1, V2, D1, D2, k)

        # align bases with base alignment matrix
        V1_rot = V1[:, 0:k]
        V2_rot = V2[:, 0:k] @ B

        # calculate correspondence matrix C
        C = calc_C_as_in_quasiharmonicpaper(Cor1, Cor2, V1_rot, V2_rot, k, q)
        # print(np.diagonal(C))

        # use eigenvectors for alignment
        G1_emb = C @ V1_rot.T  # [:, 0: k].T;

        G2_emb = V2_rot.T  # [:, 0: k].T;
    else:
        A = calc_coefficient_matrix(Cor1, V1, k, q)

        B = calc_coefficient_matrix(Cor2, V2, k, q)

        C = calc_correspondence_matrix_ortho(A, B, k)
        #C = calc_C_as_in_quasiharmonicpaper(Cor1, Cor2, V1[:,0:k], V2[:,0:k], k, q)
        # print(np.diagonal(C))

        G1_emb = C @ V1[:, 0: k].T

        G2_emb = V2[:, 0: k].T

    # matching = []

    # # matching
    # if (icp):
    #     matching = iterative_closest_point(
    #         V1_rot, V2_rot, C, icp_its, k, match, Cor1, Cor2, q)
    # else:
    #     if match == 1:
    #         matching = greedyNN(G1_emb, G2_emb)
    #     if match == 2:
    #         matching = sort_greedy(G1_emb, G2_emb)
    #     if match == 3:
    #         matching = hungarian_matching(G1_emb, G2_emb)

    #     matching = dict(matching.astype(int))

    # return matching

    return G1_emb, G2_emb


# def functional_maps(A1, A2, q, k, n_eig, laa, icp, icp_its, lower_t, upper_t, linsteps):

#     # 1:nn 2:sortgreedy 3: jv
#     match = laa

#     t = np.linspace(lower_t, upper_t, q)
#     if(not linsteps):
#         t = np.logspace(lower_t, upper_t, q)

#     n = np.shape(A1)[0]

#     D1, V1 = decompose_laplacian(A1, True, n_eig)
#   # #  print(V1[20,200])
#     D2, V2 = decompose_laplacian(A2, True, n_eig)
#    # print(V2[20, 200])

#     Cor1 = calc_corresponding_functions(n, q, t, D1, V1)
#     Cor2 = calc_corresponding_functions(n, q, t, D2, V2)

#     A = calc_coefficient_matrix(Cor1, V1, k, q)

#     B = calc_coefficient_matrix(Cor2, V2, k, q)

#     C = calc_correspondence_matrix_ortho(A, B, k)
#     #C = calc_C_as_in_quasiharmonicpaper(Cor1, Cor2, V1[:,0:k], V2[:,0:k], k, q)
#     print(np.diagonal(C))

#     G1_emb = C @ V1[:, 0: k].T

#     G2_emb = V2[:, 0: k].T

#     matching = []

#     if(icp):
#         matching = iterative_closest_point(
#             V1, V2, C, icp_its, k, match, Cor1, Cor2, q)
#     else:
#         if match == 1:
#             matching = greedyNN(G1_emb, G2_emb)
#         if match == 2:
#             matching = sort_greedy(G1_emb, G2_emb)
#         if match == 3:
#             matching = hungarian_matching(G1_emb, G2_emb)

#         matching = dict(matching.astype(int))

#     return matching


def edgelist_to_adjmatrix(edgeList_file):
    edge_list = np.loadtxt(edgeList_file, usecols=range(2)).astype(np.int)
    m = edge_list.shape[0]
    a = sps.coo_matrix((np.ones(m), (edge_list[:, 0], edge_list[:, 1])))
    a = a + a.T
    a.data = np.ones(a.data.shape)
    n = a.shape[0]
    return a  # .todense().A1.reshape(n, n) # uncomment to get the dense version


def adj_to_laplacian(mat, normalized):
    """
    Converts a sparse or dence adjacency matrix to Laplacian.

    Parameters
    ----------
    mat : obj
        Input adjacency matrix. If it is a Laplacian matrix already, return it.
    normalized : bool
        Whether to use normalized Laplacian.
        Normalized and unnormalized Laplacians capture different properties of graphs, e.g. normalized Laplacian spectrum can determine whether a graph is bipartite, but not the number of its edges. We recommend using normalized Laplacian.
    Returns
    -------
    obj
        Laplacian of the input adjacency matrix
    Examples
    --------
    >>> mat_to_laplacian(numpy.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]), False)
    [[ 2, -1, -1], [-1,  2, -1], [-1, -1,  2]]
    """
    if sps.issparse(mat):
        if np.all(mat.diagonal() >= 0):  # Check diagonal
            if np.all((mat-sps.diags(mat.diagonal())).data <= 0):  # Check off-diagonal elements
                return mat
    else:
        if np.all(np.diag(mat) >= 0):  # Check diagonal
            if np.all(mat - np.diag(mat) <= 0):  # Check off-diagonal elements
                return mat
    deg = np.squeeze(np.asarray(mat.sum(axis=1)))
    if sps.issparse(mat):
        L = sps.diags(deg) - mat
    else:
        L = np.diag(deg) - mat
    if not normalized:
        return L
    with np.errstate(divide='ignore'):
        sqrt_deg = 1.0 / np.sqrt(deg)
    sqrt_deg[sqrt_deg == np.inf] = 0
    if sps.issparse(mat):
        sqrt_deg_mat = sps.diags(sqrt_deg)
    else:
        sqrt_deg_mat = np.diag(sqrt_deg)
    return sqrt_deg_mat.dot(L).dot(sqrt_deg_mat)


def decompose_laplacian(A, normalized=True, n_eig=100):
    l = adj_to_laplacian(A, normalized)
    D, V = spsl.eigsh(l, n_eig, which='SM')
    return [D, V]


def decompose_rw_normalized_laplacian(A):

    #  adjacency matrix

    Deg = np.diag((np.sum(A, axis=1)))

    n = np.shape(Deg)[0]

    L = np.identity(n) - np.linalg.inv(Deg) @ A

   # print((sci.fractional_matrix_power(Deg, -0.5) * A * sci.fractional_matrix_power(Deg, -0.5)))
    # '[V1, D1] = eig(L1);

    D, V = np.linalg.eig(L)

    return [D, V]


def calc_corresponding_functions(n, q, t, d, V):

    # corresponding functions are the heat kernel diagonals in each time step
    # t= time steps, d= eigenvalues, V= eigenvectors, n= number of nodes, q= number of corresponding functions
    t = t[:, np.newaxis]
    d = d[:, np.newaxis]

    V_square = np.square(V)

    time_and_eigv = np.dot((d), np.transpose(t))

    time_and_eigv = np.exp(-1*time_and_eigv)

    Cores = np.dot(V_square, time_and_eigv)

    return Cores


def calc_coefficient_matrix(Corr, V, k, q):
    coefficient_matrix = np.linalg.lstsq(V[:, 0:k], Corr, rcond=None)
    # print(type(coefficient_matrix))
    return coefficient_matrix[0]


def calc_correspondence_matrix(A, B, k):
    C = np.zeros([k, k])
    At = A.T
    Bt = B.T

    for i in range(0, k):
        C[i, i] = np.linalg.lstsq(
            Bt[:, i].reshape(-1, 1), At[:, i].reshape(-1, 1), rcond=None)[0]

    return C


def calc_correspondence_matrix_ortho_diag(A, B, k):
    C = np.zeros([k, k])
    At = A.T
    Bt = B.T

    for i in range(0, k):
        C[i, i] = np.sign(np.linalg.lstsq(
            Bt[:, i].reshape(-1, 1), At[:, i].reshape(-1, 1), rcond=None)[0])

    return C


def calc_correspondence_matrix_ortho(A, B, k):
    #C = np.zeros([k,k])
    At = A.T
    Bt = B.T

    C = sci.linalg.orthogonal_procrustes(Bt, At)[0]

    C_norms = np.linalg.norm(C)

    C_normalized = normalize(C, axis=1)
    # for i in range(0,k):
    # print(np.shape(C))
    # print(C)
    # print('\n')
    # print(C_normalized)

    # print(np.sum(C_normalized,axis=1))
    #print(np.sum(C, axis=1))
    # return C_normalized
    return C_normalized


# def nearest_neighbor_matching(G1_emb, G2_emb):
#     n = np.shape(G1_emb)[1]
#     nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(G1_emb.T)
#     distances, indices = nbrs.kneighbors(G2_emb.T)
#     indices = np.c_[np.linspace(0, n-1, n).astype(int), indices.astype(int)]
#     return indices
# #


# def hungarian_matching(G1_emb, G2_emb):
#     import lapjv
#     print('hungarian_matching: calculating distance matrix')

#     dist = sci.spatial.distance_matrix(G1_emb.T, G2_emb.T)
#     n = np.shape(dist)[0]
#     # print(np.shape(dist))
#     print('hungarian_matching: calculating matching')
#     cols, rows, _ = lapjv.lapjv(dist)
#     matching = np.c_[cols, np.linspace(0, n-1, n).astype(int)]
#     matching = matching[matching[:, 0].argsort()]
#     return matching.astype(int)


# def iterative_closest_point(V1, V2, C, it, k, match, Cor1, Cor2, q):
#     G1 = V1[:, 0: k].T
#     G2_emb = V2[:, 0: k].T
#     n = np.shape(G2_emb)[1]

#     for i in range(0, it):

#         print('icp iteration '+str(i))
#         G1_emb = C@V1[:, 0:k].T
#        # print('calculating hungarian in icp')
#         M = []

#         if (match == 1):
#             M = nearest_neighbor_matching(G1_emb, G2_emb)
#         if match == 2:
#             M = sort_greedy(G1_emb, G2_emb)
#         if match == 3:
#             M = hungarian_matching(G1_emb, G2_emb)
#         G2_cur = np.zeros([k, n])
#        ## print('finding nearest neighbors in eigenvector matrix icp')
#         for j in range(0, n):

#             G2idx = M[j, 1]
#             G2_cur[:, G2idx] = G2_emb[:, j]
#        ## print('calculating correspondence matrix in icp')
#         # C=calc_correspondence_matrix(G1,G2_cur,k)
#         C = calc_correspondence_matrix_ortho(G1, G2_cur, k)
#         #calc_C_as_in_quasiharmonicpaper(Cor1, Cor2, V1[:,0:k], V2[:,0:k], k, q)
#         #C = calc_correspondence_matrix_ortho_diag(G1, G2_cur, k)
#         C_show = C
#         # C_show[C_show < 0.13] = 0.0
#        # plt.imshow(np.abs(C_show))
#        # plt.show()

#        # print('calculated correspondence matrix in icp')
#        # print('\n')
#     G1_emb = C@V1[:, 0:k].T

#     if (match == 1):
#         M = nearest_neighbor_matching(G1_emb, G2_emb)
#     if match == 2:
#         M = sort_greedy(G1_emb, G2_emb)
#     if match == 3:
#         M = hungarian_matching(G1_emb, G2_emb)
#     print('\n')
#     return dict(M.astype(int))


# def greedyNN(G1_emb, G2_emb):
#     print('greedyNN: calculating distance matrix')

#     dist = sci.spatial.distance_matrix(G1_emb.T, G2_emb.T)
#     n = np.shape(dist)[0]
#     # print(np.shape(dist))
#     print('greedyNN: calculating matching')
#     idx = np.argsort(dist, axis=0)
#     matching = np.ones([n, 1])*(n+1)
#     for i in range(0, n):
#         matched = False
#         cur_idx = 0
#         while(not matched):
#             # print([cur_idx,i])
#             if(not idx[cur_idx, i] in matching):
#                 matching[i, 0] = idx[cur_idx, i]

#                 matched = True
#             else:
#                 cur_idx += 1
#                 # print(cur_idx)

#     matching = np.c_[np.linspace(0, n-1, n).astype(int), matching]
#     return matching.astype(int)


# def sort_greedy(G1_emb, G2_emb):
#     print('sortGreedy: calculating distance matrix')

#     dist = sci.spatial.distance_matrix(G1_emb.T, G2_emb.T)
#     n = np.shape(dist)[0]
#     # print(np.shape(dist))
#     print('sortGreedy: calculating matching')
#     dist_platt = np.ndarray.flatten(dist)
#     idx = np.argsort(dist_platt)
#     k = idx//n
#     r = idx % n
#     idx_matr = np.c_[k, r]
#    # print(idx_matr)
#     G1_elements = set()
#     G2_elements = set()
#     i = 0
#     j = 0
#     matching = np.ones([n, 2])*(n+1)
#     while(len(G1_elements) < n):
#         if (not idx_matr[i, 0] in G1_elements) and (not idx_matr[i, 1] in G2_elements):
#             # print(idx_matr[i,:])
#             matching[j, :] = idx_matr[i, :]

#             G1_elements.add(idx_matr[i, 0])
#             G2_elements.add(idx_matr[i, 1])
#             j += 1
#             # print(len(G1_elements))

#         i += 1

#    # print(idx)
#     matching = np.c_[matching[:, 1], matching[:, 0]]
#     matching = matching[matching[:, 0].argsort()]
#     return matching.astype(int)


# def eval_matching(matching, gt):

#     n = float(len(gt))
#     acc = 0.0
#     for i in matching:
#         if i in gt and matching[i] == gt[i]:
#             acc += 1.0
#    # print(acc/n)
#     return acc/n


def read_regal_matrix(file):

    nx_graph = nx.read_edgelist(file, nodetype=int, comments="%")
    A = nx.adjacency_matrix(nx_graph)
    n = int(np.shape(A)[0]/2)
    A1 = A[0:n, 0:n]
    A2 = A[n:2*n, n:2*n]
    return A1.todense(), A2.todense()


def functional_maps_coupled_bases(A1, A2, q, k, laa, icp, icp_its, lower_t, upper_t, linsteps):
    # corresponding functions
    # eigenvectors
    # eigenvalues
    return 0


def calc_rotation_matrices(Cor1, Cor2, V1, V2, k, q):

    rotV1, _, rotV2 = np.linalg.svd(V1[:, 0:k].T@Cor1@Cor2.T@V2[:, 0:k])

    return rotV1, rotV2.T


def calc_C_as_in_quasiharmonicpaper(Cor1, Cor2, V1, V2, k, q):
    leftside = Cor1.T@V1[:, 0:k]
    rightside = V2[:, 0:k].T@Cor2

    left = np.diag(leftside[0, :])

    right = rightside[:, 0]
    for i in range(1, q):
        left = np.concatenate((left, np.diag(leftside[i, :])))
        right = np.concatenate((right, rightside[:, i]))

   # print(np.shape(left))
   # print(np.shape(right))

    C_diag = np.linalg.lstsq(left, right, rcond=None)[0]
   # print(C_diag)
  #  print(np.shape(C_diag))
    return np.diag(C_diag)


if __name__ == '__main__':
    # args = parse_args()
    main()
