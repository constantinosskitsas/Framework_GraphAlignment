import numpy as np
import sklearn.metrics.pairwise
import scipy.sparse as sps
import argparse
import time
from scipy.sparse import csr_matrix, coo_matrix
from sklearn.neighbors import KDTree
# from data import ReadFile
from . import unsup_align, embedding
#original code from https://github.com/GemsLab/CONE-Align

# def parse_args():
#     parser = argparse.ArgumentParser(description="Run CONE Align.")

#     # parser.add_argument('--true_align', nargs='?', default='data/synthetic-combined/arenas/arenas950-1/arenas_edges-mapping-permutation.txt',
#     #                     help='True alignment file.')
#     # parser.add_argument('--combined_graph', nargs='?',
#     #                     default='data/synthetic-combined/arenas/arenas950-1/arenas_combined_edges.txt', help='Edgelist of combined input graph.')
#     # parser.add_argument('--output_stats', nargs='?',
#     #                     default='output/stats/arenas/arenas950-1.log', help='Output path for log file.')
#     # parser.add_argument('--store_align', action='store_true',
#     #                     help='Store the alignment matrix.')
#     # parser.add_argument('--output_alignment', nargs='?',
#     #                     default='output/alignment_matrix/arenas/arenas950-1', help='Output path for alignment matrix.')

#     # Node Embedding
#     # parser.add_argument('--embmethod', nargs='?',
#     #                     default='netMF', help='Node embedding method.')
#     # netMF parameters
#     # parser.add_argument("--rank", default=256, type=int,
#     #                     help='Number of eigenpairs used to approximate normalized graph Laplacian.')

#     parser.add_argument("--dim", default=64, type=int,
#                         help='Dimension of embedding.')
#     parser.add_argument("--window", default=10, type=int,
#                         help='Context window size.')
#     parser.add_argument("--negative", default=1.0, type=float,
#                         help='Number of negative samples.')

#     # parser.add_argument('--store_emb', action='store_true',
#     #                     help='Store the node embedding.')
#     # parser.add_argument('--embeddingA', nargs='?', default='emb/netMF/arenas/arenas950-1.graph1.npy',
#     #                     help='Node embedding path for the first graph.')
#     # parser.add_argument('--embeddingB', nargs='?', default='emb/netMF/arenas/arenas950-1.graph2.npy',
#     #                     help='Node embedding path for the second graph.')

#     # Embedding Space Alignment
#     # convex initialization parameters
#     parser.add_argument('--niter_init', type=int,
#                         default=10, help='Number of iterations.')
#     parser.add_argument('--reg_init', type=float, default=1.0,
#                         help='Regularization parameter.')
#     # WP optimization parameters
#     parser.add_argument('--nepoch', type=int, default=5,
#                         help='Number of epochs.')
#     parser.add_argument('--niter_align', type=int,
#                         default=10, help='Iterations per epoch.')
#     parser.add_argument('--reg_align', type=float,
#                         default=0.05, help='Regularization parameter.')
#     parser.add_argument('--bsz', type=int, default=10, help='Batch size.')
#     parser.add_argument('--lr', type=float, default=1.0, help='Learning rate.')

#     # Matching Nodes
#     parser.add_argument('--embsim', nargs='?', default='euclidean',
#                         help='Metric for comparing embeddings.')
#     parser.add_argument('--alignmethod', nargs='?',
#                         default='greedy', help='Method to align embeddings.')
#     parser.add_argument('--numtop', type=int, default=10,
#                         help='Number of top similarities to compute with kd-tree.  If None, computes all pairwise similarities.')

#     return parser.parse_known_args()[0]


def align_embeddings(embed1, embed2, CONE_args, adj1=None, adj2=None, struc_embed=None, struc_embed2=None):
    # Step 2: Align Embedding Spaces
    corr = None
    if struc_embed is not None and struc_embed2 is not None:
        if CONE_args['embsim'] == "cosine":
            corr = sklearn.metrics.pairwise.cosine_similarity(embed1, embed2)
        else:
            corr = sklearn.metrics.pairwise.euclidean_distances(embed1, embed2)
            corr = np.exp(-corr)

        # Take only top correspondences
        matches = np.zeros(corr.shape)
        matches[np.arange(corr.shape[0]), np.argmax(corr, axis=1)] = 1
        corr = matches

    # Convex Initialization
    if adj1 is not None and adj2 is not None:
        if not sps.issparse(adj1):
            adj1 = sps.csr_matrix(adj1)
        if not sps.issparse(adj2):
            adj2 = sps.csr_matrix(adj2)
        init_sim, corr_mat = unsup_align.convex_init_sparse(
            embed1, embed2, K_X=adj1, K_Y=adj2, apply_sqrt=False, niter=CONE_args['niter_init'], reg=CONE_args['reg_init'], P=corr)
    else:
        init_sim, corr_mat = unsup_align.convex_init(
            embed1, embed2, apply_sqrt=False, niter=CONE_args['niter_init'], reg=CONE_args['reg_init'], P=corr)
    # print(corr_mat)
    # print(np.max(corr_mat, axis=0))
    # print(np.max(corr_mat, axis=1))

    # Stochastic Alternating Optimization
    dim_align_matrix, corr_mat = unsup_align.align(
        embed1, embed2, init_sim, lr=CONE_args['lr'], bsz=CONE_args['bsz'], nepoch=CONE_args['nepoch'], niter=CONE_args['niter_align'], reg=CONE_args['reg_align'])
    # print(dim_align_matrix.shape, corr_mat.shape)

    # Step 3: Match Nodes with Similar Embeddings
    # Align embedding spaces
    aligned_embed1 = embed1.dot(dim_align_matrix)
    # Greedily match nodes
    # greedily align each embedding to most similar neighbor
    # if CONE_args['alignmethod'] == 'greedy':
    #     # KD tree with only top similarities computed
    #     if CONE_args['numtop'] is not None:
    alignment_matrix = kd_align(
        aligned_embed1, embed2, distance_metric=CONE_args['embsim'], num_top=CONE_args['numtop'])
    # # All pairwise distance computation
    # else:
    #     if CONE_args['embsim'] == "cosine":
    #         alignment_matrix = sklearn.metrics.pairwise.cosine_similarity(
    #             aligned_embed1, embed2)
    #     else:
    #         alignment_matrix = sklearn.metrics.pairwise.euclidean_distances(
    #             aligned_embed1, embed2)
    #         alignment_matrix = np.exp(-alignment_matrix)
    return alignment_matrix, sklearn.metrics.pairwise.euclidean_distances(aligned_embed1, embed2)


# def get_counterpart(alignment_matrix, true_alignments):
#     n_nodes = alignment_matrix.shape[0]

#     correct_nodes = []
#     counterpart_dict = {}

#     if not sps.issparse(alignment_matrix):
#         sorted_indices = np.argsort(alignment_matrix)

#     for node_index in range(n_nodes):
#         # default: assume identity mapping, and the node should be aligned to itself
#         target_alignment = node_index
#         # if we have true alignments (which we require), use those for each node
#         if true_alignments is not None:
#             target_alignment = int(true_alignments[node_index])
#         if sps.issparse(alignment_matrix):
#             row, possible_alignments, possible_values = sps.find(
#                 alignment_matrix[node_index])
#             node_sorted_indices = possible_alignments[possible_values.argsort(
#             )]
#         else:
#             node_sorted_indices = sorted_indices[node_index]
#         if target_alignment in node_sorted_indices[-1:]:
#             correct_nodes.append(node_index)
#         counterpart = node_sorted_indices[-1]
#         counterpart_dict[node_index] = counterpart

#     return correct_nodes, counterpart_dict


def kd_align(emb1, emb2, normalize=False, distance_metric="euclidean", num_top=10):
    kd_tree = KDTree(emb2, metric=distance_metric)

    row = np.array([])
    col = np.array([])
    data = np.array([])

    dist, ind = kd_tree.query(emb1, k=num_top)
    # print("queried alignments")
    row = np.array([])
    for i in range(emb1.shape[0]):
        row = np.concatenate((row, np.ones(num_top) * i))
    col = ind.flatten()
    data = np.exp(-dist).flatten()
    sparse_align_matrix = coo_matrix(
        (data, (row, col)), shape=(emb1.shape[0], emb2.shape[0]))
    return sparse_align_matrix.tocsr()


def main(data, **args):

    Src = data['Src']
    Tar = data['Tar']

    # min_dim = min(Src.shape[0] - 1, Tar.shape[0] - 1)
    # if args['dim'] > min_dim:
    #     args['dim'] = min_dim

    if args['dim'] > Src.shape[0] - 1:
        args['dim'] = Src.shape[0] - 1

    # global args
    # args = parse_args()

    # node_num = int(adj.shape[0] / 2)
    # Tar = adj[:node_num, :node_num]
    # Src = adj[node_num:, node_num:]
    # Tar= ReadFile.edgelist_to_adjmatrix1("data/noise_level_1/arenas_orig.txt")
    # Src = ReadFile.edgelist_to_adjmatrix1("data/noise_level_1/edges_4.txt")
    start = time.time()
   # step1: obtain normalized proximity-preserving node embeddings
    # if (args.embmethod == "netMF"):
    print("0")
    emb_matrixA = embedding.netmf(
        Src, dim=args['dim'], window=args['window'], b=args['negative'], normalize=True)
    print("A")
    emb_matrixB = embedding.netmf(
        Tar, dim=args['dim'], window=args['window'], b=args['negative'], normalize=True)
    print("B")

    # step2 and 3: align embedding spaces and match nodes with similar embeddings
    alignment_matrix, cost_matrix = align_embeddings(
        emb_matrixA,
        emb_matrixB,
        args,
        adj1=csr_matrix(Src),
        adj2=csr_matrix(Tar),
        struc_embed=None,
        struc_embed2=None
    )
    total_time = time.time() - start
    print(("time for CONE-align (in seconds): %f" % total_time))

    return alignment_matrix, cost_matrix
