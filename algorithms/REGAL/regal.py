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
    return parser.parse_known_args()[0]


def main(adj) -> object:
    global REGAL_args
    REGAL_args = parse_args()

    if REGAL_args.attributes is not None:
        # load vector of attributes in from file
        REGAL_args.attributes = np.load(REGAL_args.attributes)
    embed = learn_representations(adj)
    emb1, emb2 = get_embeddings(embed)
    if REGAL_args.numtop == 0:
        REGAL_args.numtop = None
    alignment_matrix = get_embedding_similarities(
        emb1, emb2, num_top=REGAL_args.numtop)
    return alignment_matrix


# Should take in a file with the input graph as edgelist (REGAL_args.input)
# Should save representations to REGAL_args.output
def learn_representations(adj):
    graph = Graph(adj, node_attributes=REGAL_args.attributes)
    max_layer = REGAL_args.untillayer
    if REGAL_args.untillayer == 0:
        max_layer = None
    alpha = REGAL_args.alpha
    num_buckets = REGAL_args.buckets  # BASE OF LOG FOR LOG SCALE
    if num_buckets == 1:
        num_buckets = None
    rep_method = RepMethod(max_layer=max_layer,
                           alpha=alpha,
                           k=REGAL_args.k,
                           num_buckets=num_buckets,
                           normalize=True,
                           gammastruc=REGAL_args.gammastruc,
                           gammaattr=REGAL_args.gammaattr)
    if max_layer is None:
        max_layer = 1000
    representations = xnetmf.get_representations(graph, rep_method)
    return representations


# pickle.dump(representations, open(REGAL_args.output, "w"))


def recovery(gt1, mb):
    nodes = len(gt1)
    count = 0
    for i in range(nodes):
        if gt1[i] == mb[i]:
            count = count + 1
    return count / nodes


# REGAL_args = parse_args()
