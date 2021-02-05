from random import random
import networkx as nx
import numpy as np
import argparse
import networkx as nx
import time
import os
import sys

try:
    import cPickle as pickle
except ImportError:
    import pickle
from scipy.sparse import csr_matrix

from . import xnetmf
from .config import RepMethod, Graph
from .alignments import get_embeddings, get_embedding_similarities


def parse_args():
    parser = argparse.ArgumentParser(description="Run REGAL.")

    parser.add_argument('--input', nargs='?', default='data/arenas_combined_edges2.txt',
                        help="Edgelist of combined input graph")

    parser.add_argument('--output', nargs='?', default='emb/arenas990-1.emb',
                        help='Embeddings path')

    parser.add_argument('--attributes', nargs='?', default=None,
                        help='File with saved numpy matrix of node attributes, or int of number of attributes to synthetically generate.  Default is 5 synthetic.')

    parser.add_argument('--attrvals', type=int, default=2,
                        help='Number of attribute values. Only used if synthetic attributes are generated')

    parser.add_argument('--dimensions', type=int, default=128,
                        help='Number of dimensions. Default is 128.')

    parser.add_argument('--k', type=int, default=10,
                        help='Controls of landmarks to sample. Default is 10.')

    parser.add_argument('--untillayer', type=int, default=2,
                        help='Calculation until the layer for xNetMF.')
    parser.add_argument('--alpha', type=float, default=0.01,
                        help="Discount factor for further layers")
    parser.add_argument('--gammastruc', type=float, default=1,
                        help="Weight on structural similarity")
    parser.add_argument('--gammaattr', type=float, default=1,
                        help="Weight on attribute similarity")
    parser.add_argument('--numtop', type=int, default=10,
                        help="Number of top similarities to compute with kd-tree.  If 0, computes all pairwise similarities.")
    parser.add_argument('--buckets', default=2, type=float,
                        help="base of log for degree (node feature) binning")
    return parser.parse_args()


def main(adj, args: object) -> object:
    if args.attributes is not None:
        # load vector of attributes in from file
        args.attributes = np.load(args.attributes)
    embed = learn_representations(args, adj)
    emb1, emb2 = get_embeddings(embed)
    if args.numtop == 0:
        args.numtop = None
    alignment_matrix = get_embedding_similarities(
        emb1, emb2, num_top=args.numtop)
    return alignment_matrix


# Should take in a file with the input graph as edgelist (args.input)
# Should save representations to args.output
def learn_representations(args, adj):
    graph = Graph(adj, node_attributes=args.attributes)
    max_layer = args.untillayer
    if args.untillayer == 0:
        max_layer = None
    alpha = args.alpha
    num_buckets = args.buckets  # BASE OF LOG FOR LOG SCALE
    if num_buckets == 1:
        num_buckets = None
    rep_method = RepMethod(max_layer=max_layer,
                           alpha=alpha,
                           k=args.k,
                           num_buckets=num_buckets,
                           normalize=True,
                           gammastruc=args.gammastruc,
                           gammaattr=args.gammaattr)
    if max_layer is None:
        max_layer = 1000
    representations = xnetmf.get_representations(graph, rep_method)
    return representations


# pickle.dump(representations, open(args.output, "w"))


def recovery(gt1, mb):
    nodes = len(gt1)
    count = 0
    for i in range(nodes):
        if gt1[i] == mb[i]:
            count = count + 1
    return count / nodes


if __name__ == "__main__":
    args = parse_args()
    hi = 0
    mb = main(args)
    mb = mb + 1
    gmb = gt("data/noise_level_1/gt_1.txt")
    gmb = gmb + 1
