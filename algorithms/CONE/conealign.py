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

    dim_align_matrix, corr_mat = unsup_align.align(
        embed1, embed2, init_sim, lr=CONE_args['lr'], bsz=CONE_args['bsz'], nepoch=CONE_args['nepoch'], niter=CONE_args['niter_align'], reg=CONE_args['reg_align'])
 
    aligned_embed1 = embed1.dot(dim_align_matrix)

    alignment_matrix = kd_align(
        aligned_embed1, embed2, distance_metric=CONE_args['embsim'], num_top=CONE_args['numtop'])

    return alignment_matrix, sklearn.metrics.pairwise.euclidean_distances(aligned_embed1, embed2)
def align_embeddings1(embed1, embed2, CONE_args, adj1=None, adj2=None, struc_embed=None, struc_embed2=None):
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

    # Stochastic Alternating Optimization
    dim_align_matrix, corr_mat = unsup_align.align(
        embed1, embed2, init_sim, lr=CONE_args['lr'], bsz=CONE_args['bsz'], nepoch=CONE_args['nepoch'], niter=CONE_args['niter_align'], reg=CONE_args['reg_align'])
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
    print("Cone")
    Src = data['Src']
    Tar = data['Tar']

    #min_dim = min(Src.shape[0] - 1, Tar.shape[0] - 1)
    #if args['dim'] > min_dim:
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
    emb_matrixA = embedding.netmf(
        Src, dim=args['dim'], window=args['window'], b=args['negative'], normalize=True)

    emb_matrixB = embedding.netmf(
        Tar, dim=args['dim'], window=args['window'], b=args['negative'], normalize=True)
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
