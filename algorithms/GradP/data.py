import networkx as nx


import torch_geometric.utils.convert as cv
from torch_geometric.data import NeighborSampler as RawNeighborSampler

import pandas as pd
from  algorithms.GradP.utils import *
import warnings
import argparse
warnings.filterwarnings('ignore')
import collections
import networkx as nx
import copy 
from sklearn.metrics import roc_auc_score
import os
from  algorithms.GradP.models import *
import numpy as np
import random
import torch
from functools import reduce


'''
Please note that some function codes are apopted and revised from
https://github.com/vinhsuhi/GAlign
https://github.com/deepopo/CENALP
'''

def na_dataloader(args):
    
    if args.mode == 'not_perturbed': 
        # for douban dataset
        G1, G2= loadG(args.data_folder, args.graphname)           
      
        if args.graphname == 'am-td': # will be combined soon
        
            source_dataset = Dataset('dataset\\DataProcessing\\allmv_tmdb\\allmv')
            target_dataset = Dataset('dataset\\DataProcessing\\allmv_tmdb\\tmdb')
            gt_dr = 'dataset\\DataProcessing\\allmv_tmdb\\dictionaries\\groundtruth'
            gt_dict = graph_utils.load_gt(gt_dr, source_dataset.id2idx, target_dataset.id2idx, 'dict')
            gt_dict = DeleteDuplicatedElement(gt_dict)
    
            attr1, attr2, attribute_sim  = AttributeProcessing(args, G1, G2, gt_dict)
            G2, alignment_dict, alignment_dict_reversed = preprocessing(G1, G2, gt_dict)
            idx1_dict, idx2_dict = create_idx_dict_pair(G1,G2,alignment_dict)
            
            return G1, G2, attr1, attr2, alignment_dict, alignment_dict_reversed, idx1_dict, idx2_dict
        
        elif args.graphname == 'fl-my': # will be combined soon
        
            source_dataset = Dataset('dataset\\DataProcessing\\fl-my\\flickr')
            target_dataset = Dataset('dataset\\DataProcessing\\fl-my\\myspace')
            gt_dr = 'dataset\\DataProcessing\\fl-my\\dictionaries\\groundtruth'
            gt_dict = graph_utils.load_gt(gt_dr, source_dataset.id2idx, target_dataset.id2idx, 'dict')
    
            attr1, attr2, attribute_sim  = AttributeProcessing(args, G1, G2, gt_dict)
            G2, alignment_dict, alignment_dict_reversed = preprocessing(G1, G2, gt_dict)
            idx1_dict, idx2_dict = create_idx_dict_pair(G1,G2,alignment_dict)
            
            return G1, G2, attr1, attr2, alignment_dict, alignment_dict_reversed, idx1_dict, idx2_dict

            
        else:        
            
            alignment_dict, alignment_dict_reversed = read_alignment(args.alignment_folder, args.graphname)
            attr1, attr2, attribute_sim = AttributeProcessing(args, G1, G2, alignment_dict)
            G2, alignment_dict, alignment_dict_reversed = preprocessing(G1, G2, alignment_dict)
            idx1_dict, idx2_dict = create_idx_dict_pair(G1,G2,alignment_dict)
            
            return G1, G2, attr1, attr2, alignment_dict, alignment_dict_reversed, idx1_dict, idx2_dict

            
    else: 
        # for perturbed
        G1, G2 = loadG(args.data_folder, args.graphname)
        alignment_dict, alignment_dict_reversed = read_alignment(args.alignment_folder, args.graphname)
        G3, G4, Ggt = PerturbedProcessing(G1, G2, 0, args.edge_portion, args.graphname)
        
        attr1, attr2, attribute_sim = AttributeProcessing(args, G3, G4, alignment_dict)
        attr1 = random_flipping_att(attr1, args.att_portion)
        attr2 = random_flipping_att(attr2, args.att_portion)
        #alignment_dict, alignment_dict_reversed = read_alignment(args.alignment_folder, args.graphname)  
        idx1_dict, idx2_dict = create_idx_dict_pair(G3,G4,alignment_dict)
        
        return G3, G4, attr1, attr2, alignment_dict, alignment_dict_reversed, idx1_dict, idx2_dict
    
    
'''Data preprocessing'''

def Reordering(G1,G2):
    G1 = nx.convert_node_labels_to_integers(G1, first_label=1, ordering='default', label_attribute=None)
    G2 = nx.convert_node_labels_to_integers(G2, first_label=G1.number_of_nodes()+1, ordering='default', label_attribute=None)
    return G1,G2

def ReorderingSame(G1,G2):
    G1 = nx.convert_node_labels_to_integers(G1, first_label=1, ordering='default', label_attribute=None)
    G2 = nx.convert_node_labels_to_integers(G2, first_label=1, ordering='default', label_attribute=None)
    return G1,G2


def PerturbedProcessing(G1, G2, com_portion, rand_portion, graphname):

    G3 = copy.deepcopy(G1)
    #Groundtruth graph
    Ggt = copy.deepcopy(G1)
    #Input 2 graphs
    G3, G4 = perturb_edge_pair(G3, rand_portion)
    G3, G4 = ReorderingSame(G3,G4) #0505 disabled
    
    #export to the files
    nx.write_edgelist(G3, "{}1_ran{}.edges".format(graphname, rand_portion),delimiter=',',data=False)
    nx.write_edgelist(G4, "{}2_ran{}.edges".format(graphname, rand_portion),delimiter=',',data=False)
    print('exporting data complete')
    
    return G3, G4, Ggt


    
def AttributeProcessing(args,G1,G2, alignment_dict):

    attribute_sim, attr1, attr2, attr1_pd, attr2_pd = \
        read_attribute(args.attribute_folder, args.graphname, G1, G2, alignment_dict)

    featpd1 = pd.DataFrame(attr1)
    featpd2 = pd.DataFrame(attr2)
    featnpy1 = featpd1.to_numpy() 
    featnpy2 = featpd2.to_numpy() 
    # print(featpd1)
    # np.save("feats1.npy",featnpy1)
    # np.save("feats2.npy",featnpy2)
    # print("feats exporting complete")
    
    return attr1, attr2, attribute_sim





''' perturbation '''
def perturb_edge_pair(G, rand_portion = 0.1):
    
    G_copy = copy.deepcopy(G)
    
    edgelist = list(G.edges)
    
    num_mask_rand = int(len(edgelist)*rand_portion)
    if rand_portion == 0:
        return G, G_copy
    
    for _ in range(num_mask_rand):
        e = sample(list(edgelist),1)
        
        start_vertex = e[0][0]
        end_vertex = e[0][1]
        
        if G.degree[start_vertex] >= 2 and G.degree[end_vertex] >= 2:
            G.remove_edges_from(e)
            
    for _ in range(num_mask_rand):
        e = sample(list(edgelist),1)
        
        start_vertex = e[0][0]
        end_vertex = e[0][1]
        
        if G_copy.degree[start_vertex] >= 2 and G_copy.degree[end_vertex] >= 2:
            G_copy.remove_edges_from(e)
            
    return G, G_copy

def perturb_edge_pair_oneside(G, rand_portion = 0.1):
    
    G_copy = copy.deepcopy(G)
    
    edgelist = list(G.edges)
    
    num_mask_rand = int(len(edgelist)*rand_portion)
            
    for _ in range(num_mask_rand):
        e = sample(list(edgelist),1)
        
        start_vertex = e[0][0]
        end_vertex = e[0][1]
        
        if G_copy.degree[start_vertex] >= 2 and G_copy.degree[end_vertex] >= 2:
            G_copy.remove_edges_from(e)
            
    return G, G_copy


def perturb_edge_pair_real(G1, G2, dictionary, com_portion = 0.1, rand_portion = 0.05):
    
    edgelist1 = list(G1.edges)
    edgelist2 = list(G2.edges)
    
    edgelist_com = list(set(edgelist1) & set(edgelist2))
    
    num_mask_common = int(len(edgelist2)*com_portion)
    num_mask_rand = int(len(edgelist1)*rand_portion)
    
    for _ in range(num_mask_common):
        e = sample(list(edgelist1),1)
   
        start_vertex = e[0][0]
        end_vertex = e[0][1]
        
        try:
            e2 = [(dictionary[start_vertex], dictionary[end_vertex])]
        except:
            pass
        
        if G1.degree[start_vertex] >= 2 and G1.degree[end_vertex] >= 2:
            G1.remove_edges_from(e)
            try:
                G2.remove_edges_from(e2)
            except:
                pass
            
    for _ in range(num_mask_rand):
        e = sample(list(edgelist1),1)
        
        start_vertex = e[0][0]
        end_vertex = e[0][1]
        
        if G1.degree[start_vertex] >= 2 and G1.degree[end_vertex] >= 2:
            G1.remove_edges_from(e)
            
    for _ in range(num_mask_rand):
        e = sample(list(edgelist2),1)
        
        start_vertex = e[0][0]
        end_vertex = e[0][1]
        
        if G2.degree[start_vertex] >= 2 and G2.degree[end_vertex] >= 2:
            G2.remove_edges_from(e)
            
    return G1, G2

class Dataset:
    """
    this class are copied from the repo: https://github.com/vinhsuhi/GAlign
    this class receives input from graphsage format with predefined folder structure, the data folder must contains these files:
    G.json, id2idx.json, features.npy (optional)

    Arguments:
    - data_dir: Data directory which contains files mentioned above.
    """

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self._load_id2idx()
        self._load_G()
        self._load_features()
        graph_utils.construct_adjacency(self.G, self.id2idx, sparse=False, file_path=self.data_dir + "/edges.edgelist")
        # self.load_edge_features()
        print("Dataset info:")
        print("- Nodes: ", len(self.G.nodes()))
        print("- Edges: ", len(self.G.edges()))

    def _load_G(self):
        G_data = json.load(open(os.path.join(self.data_dir, "G.json")))
        G_data['links'] = [{'source': self.idx2id[G_data['links'][i]['source']], 'target': self.idx2id[G_data['links'][i]['target']]} for i in range(len(G_data['links']))]
        self.G = json_graph.node_link_graph(G_data)


    def _load_id2idx(self):
        id2idx_file = os.path.join(self.data_dir, 'id2idx.json')
        self.id2idx = json.load(open(id2idx_file))
        self.idx2id = {v:k for k,v in self.id2idx.items()}


    def _load_features(self):
        self.features = None
        feats_path = os.path.join(self.data_dir, 'feats.npy')
        if os.path.isfile(feats_path):
            self.features = np.load(feats_path)
        else:
            self.features = None
        return self.features

    def load_edge_features(self):
        self.edge_features= None
        feats_path = os.path.join(self.data_dir, 'edge_feats.mat')
        if os.path.isfile(feats_path):
            edge_feats = loadmat(feats_path)['edge_feats']
            self.edge_features = np.zeros((len(edge_feats[0]),
                                           len(self.G.nodes()),
                                           len(self.G.nodes())))
            for idx, matrix in enumerate(edge_feats[0]):
                self.edge_features[idx] = matrix.toarray()
        else:
            self.edge_features = None
        return self.edge_features

    def get_adjacency_matrix(self, sparse=False):
        return graph_utils.construct_adjacency(self.G, self.id2idx, sparse=False, file_path=self.data_dir + "/edges.edgelist")

    def get_nodes_degrees(self):
        return graph_utils.build_degrees(self.G, self.id2idx)

    def get_nodes_clustering(self):
        return graph_utils.build_clustering(self.G, self.id2idx)

    def get_edges(self):
        return graph_utils.get_edges(self.G, self.id2idx)

    def check_id2idx(self):
        # print("Checking format of dataset")
        for i, node in enumerate(self.G.nodes()):
            if (self.id2idx[node] != i):
                print("Failed at node %s" % str(node))
                return False
        # print("Pass")
        return True
''' file loading '''
def read_alignment(alignment_folder, filename):
    alignment = pd.read_csv(alignment_folder + filename + '.csv', header = None)
    alignment_dict = {}
    alignment_dict_reversed = {}
    for i in range(len(alignment)):
        alignment_dict[alignment.iloc[i, 0]] = alignment.iloc[i, 1]
        alignment_dict_reversed[alignment.iloc[i, 1]] = alignment.iloc[i, 0]
    return alignment_dict, alignment_dict_reversed

def get_reversed(alignment_dict):
    alignment_dict_reversed = {}
    reversed_dictionary = {value : key for (key, value) in alignment_dict.items()}
    return alignment_dict, reversed_dictionary

def read_attribute(attribute_folder, filename, G1, G2, alignment_dict):
    try:
        attribute, attr1, attr2, attr1_pd, attr2_pd = load_attribute(attribute_folder, filename, G1, G2, alignment_dict)
       # attribute, attr1, attr2, attr1_pd, attr2_pd = load_attribute_for_perturb(attribute_folder, filename, G1, G2,alignment_dict, 0.2)
        attribute = attribute.transpose()
    except:
        attr1 = []
        attr2 = []
        attr1_pd = []
        attr2_pd = []
        attribute = []
        print('Attribute files not found.')
    return attribute, attr1, attr2, attr1_pd, attr2_pd




def load_gt(path, id2idx_src=None, id2idx_trg=None, format='matrix'):    
    if id2idx_src:
        conversion_src = type(list(id2idx_src.keys())[0])
        conversion_trg = type(list(id2idx_trg.keys())[0])
    if format == 'matrix':
        # Dense
        """
        gt = np.zeros((len(id2idx_src.keys()), len(id2idx_trg.keys())))
        with open(path) as file:
            for line in file:
                src, trg = line.strip().split()                
                gt[id2idx_src[conversion_src(src)], id2idx_trg[conversion_trg(trg)]] = 1
        return gt
        """
        # Sparse
        row = []
        col = []
        val = []
        with open(path) as file:
            for line in file:
                src, trg = line.strip().split()
                row.append(id2idx_src[conversion_src(src)])
                col.append(id2idx_trg[conversion_trg(trg)])
                val.append(1)
        gt = csr_matrix((val, (row, col)), shape=(len(id2idx_src), len(id2idx_trg)))
    else:
        gt = {}
        with open(path) as file:
            for line in file:
                src, trg = line.strip().split()
                # print(src, trg)
                if id2idx_src:
                    gt[id2idx_src[conversion_src(src)]] = id2idx_trg[conversion_trg(trg)]
                else:
                    gt[str(src)] = str(trg)
    return gt


def loadG(data_folder, filename):

    G1 = nx.Graph()
    G2 = nx.Graph()
    G1_edges = pd.read_csv(data_folder + filename + '1.edges', names = ['0', '1'])
    G1.add_edges_from(np.array(G1_edges))
    G2_edges = pd.read_csv(data_folder + filename + '2.edges', names = ['0', '1'])
    G2.add_edges_from(np.array(G2_edges))
    
    return G1, G2

def loadG_link(data_folder, test_frac, filename):
    G1 = nx.Graph()
    G2 = nx.Graph()
    G1_edges = pd.read_csv(data_folder + filename + '1.edges', names = ['0', '1'])
    G1.add_edges_from(np.array(G1_edges))
    G2_edges = pd.read_csv(data_folder + filename + '2_' + str(test_frac) + '.edges', names = ['0', '1'])
    G2.add_edges_from(np.array(G2_edges))
    test_edges = pd.read_csv(data_folder + filename + '2_' + str(test_frac) + '_test.edges', names = ['0', '1'])
    return G1, G2, test_edges

def load_attribute(attribute_folder, filename, G1, G2, alignment_dict):

    G1_nodes = list(G1.nodes()) 
    G2_nodes = list(G2.nodes())
    
    attribute1_pd = pd.read_csv(attribute_folder + filename + 'attr1.csv', header = None, index_col = 0)
    attribute2_pd = pd.read_csv(attribute_folder + filename + 'attr2.csv', header = None, index_col = 0)

    attribute1 = np.array(attribute1_pd.loc[G1_nodes, :])
    attribute2 = np.array(attribute2_pd.loc[G2_nodes, :])   

    attr_cos = cosine_similarity(attribute1, attribute2)
    
    return attr_cos, attribute1, attribute2, attribute1_pd, attribute2_pd


def load_attribute_for_perturb(attribute_folder, filename, G1, G2, alignment_dict, percent):
    
    G1_nodes = list(G1.nodes()) 
    G2_nodes = list(G2.nodes())
    noise = '_' + str(percent)

    attribute1_pd = pd.read_csv(attribute_folder +'noise/' + filename + noise + 'attr1.csv', header = None, index_col = 0)
    attribute2_pd = pd.read_csv(attribute_folder + 'noise/'+ filename + noise + 'attr2.csv', header = None, index_col = 0)
    #print("sadasdas",attribute2)

    attribute1 = np.array(attribute1_pd.loc[G1_nodes, :])
    attribute2 = np.array(attribute2_pd.loc[G2_nodes, :])   

    return attr_cos, attribute1, attribute2, attribute1_pd, attribute2_pd

def preprocessing(G1, G2, alignment_dict):
    '''
    Parameters
    ----------
    G1 : source graph
    G2 : target graph
    alignment_dict : grth dict

    '''
    # shift index for constructing union
    # construct shifted dict
    shift = G1.number_of_nodes()
    G2_list = list(G2.nodes())
    G2_shiftlist = list(idx + shift for idx in list(G2.nodes()))
    shifted_dict = dict(zip(G2_list,G2_shiftlist))
    
    #relable idx for G2
    G2 = nx.relabel_nodes(G2, shifted_dict)
    
    #update alignment dict
    align1list = list(alignment_dict.keys())
    align2list = list(alignment_dict.values())   
    shifted_align2list = [a+shift for a in align2list]
    
    groundtruth_dict = dict(zip(align1list, shifted_align2list))
    groundtruth_dict, groundtruth_dict_reversed = get_reversed(groundtruth_dict)
    
    return G2, groundtruth_dict, groundtruth_dict_reversed


def create_idx_dict_pair(G1,G2,alignment_dict):
    '''
    Make sure that this function is followed after preprocessing dict.

    '''
    
    G1list = list(G1.nodes())
    #G1list.sort()
    idx1_list = list(range(G1.number_of_nodes()))
    #make dict for G1
    idx1_dict = {a : b for b, a in zip(idx1_list,G1list)}

    
    G2list = list(G2.nodes())
    #G2list.sort()
    idx2_list = list(range(G2.number_of_nodes()))
    #make dict for G2
    idx2_dict = {c : d for d, c in zip(idx2_list,G2list)}
    
    return idx1_dict, idx2_dict


def create_idx_dict_pair_backup(G1,G2,alignment_dict):
    '''
    Make sure that this function is followed after preprocessing dict.

    '''
    
    G1list = list(G1.nodes())
    #G1list.sort()
    idx1_list = list(range(G1.number_of_nodes()))
    #make dict for G1
    idx1_dict = {a : b for b, a in zip(idx1_list,G1list)}

    
    G2list = list(G2.nodes())
    #G2list.sort()
    idx2_list = list(range(G2.number_of_nodes()))
    #make dict for G2
    idx2_dict = {c : d for d, c in zip(idx2_list,G2list)}
    
    return idx1_dict, idx2_dict

def augment_attr(Gs, Gt, attr_s, attr_t, interval):
    
    # Gs, Gt 중 max deg를 측정
    max_deg = max(max(Gs.degree, key=lambda x: x[1])[1], max(Gt.degree, key=lambda x: x[1])[1])
    print(f"max degree is {max_deg}")
    # interval을 기준으로 했을 때, 늘어나는 속성 개수를 계산
    num_attr = math.ceil(max_deg / interval)
    # n_s * num_attr 의 속성 벡터를 0 value로 초기화
    init_np_s = np.zeros((Gs.number_of_nodes(), num_attr))
    init_np_t = np.zeros((Gt.number_of_nodes(), num_attr))

    # 각 노드마다 deg를 측정하여 attr를 init_np에 assign
    for idx_s, node_s in enumerate(Gs.nodes()):
        deg_node = Gs.degree[node_s]
        init_np_s[idx_s, int(deg_node / interval) - 1] = 1

    for idx_t, node_t in enumerate(Gt.nodes()):
        deg_node = Gt.degree[node_t]
        init_np_t[idx_t, int(deg_node / interval) - 1] = 1

    # assign이 완료된 매트릭스를 기존 attr에 attach (np.append(a,b, axis =1))
    new_attr_s = np.append(attr_s, init_np_s, axis=1)
    new_attr_t = np.append(attr_t, init_np_t, axis=1)
    new_attr_s = init_np_s
    new_attr_t = init_np_t
    # 만약 len(attr_s) == 1 (plain network) 의 경우, 그냥 attr을 대체하면 된다.
    if len(attr_s) == 1:
        new_attr_s = new_attr_s[:, 1:]
        new_attr_t = new_attr_t[:, 1:]

    return new_attr_s, new_attr_t

def augment_attr_khop(Gs, Gt, attr_s, attr_t, interval, k):
    # attribute index의 순서는 G*.nodes()를 출력했을때 나오는 순서와 같음 !!
    Gs_nodes = list(Gs.nodes())
    Gt_nodes = list(Gt.nodes())
    # node: khop nbr의 구성을 갖는 dict 생성
    khopdict_source = {key : len(nx.single_source_shortest_path_length(Gs, source = key, cutoff=k)) for key in Gs_nodes}
    khopdict_target = {key : len(nx.single_source_shortest_path_length(Gt, source = key, cutoff=k)) for key in Gt_nodes}
    # Gs, Gt 중 max deg를 측정
    max_deg = max(max(khopdict_source.values()), max(khopdict_target.values()))
    print(f"max k-hop degree is {max_deg}")
    # interval을 기준으로 했을 때, 늘어나는 속성 개수를 계산
    num_attr = math.ceil(max_deg / interval)
    # n_s * num_attr 의 속성 벡터를 0 value로 초기화
    init_np_s = np.zeros((Gs.number_of_nodes(), num_attr))
    init_np_t = np.zeros((Gt.number_of_nodes(), num_attr))

    # 각 노드마다 deg를 측정하여 attr를 init_np에 assign
    for idx_s, node_s in enumerate(Gs.nodes()):
        deg_node = khopdict_source[node_s]
        init_np_s[idx_s, int(deg_node / interval) - 1] = 1

    for idx_t, node_t in enumerate(Gt.nodes()):
        deg_node = khopdict_target[node_t]
        init_np_t[idx_t, int(deg_node / interval) - 1] = 1

    # assign이 완료된 매트릭스를 기존 attr에 attach (np.append(a,b, axis =1))
    new_attr_s = np.append(attr_s, init_np_s, axis=1)
    new_attr_t = np.append(attr_t, init_np_t, axis=1)
    # new_attr_s = init_np_s
    # new_attr_t = init_np_t
    # 만약 len(attr_s) == 1 (plain network) 의 경우, 그냥 attr을 대체하면 된다.
    if len(attr_s) == 1:
        new_attr_s = new_attr_s[:, 1:]
        new_attr_t = new_attr_t[:, 1:]

    return new_attr_s, new_attr_t


def aug_trimming(aug_s, aug_t):
    #concat two matrice
    concat_aug = np.concatenate((aug_s, aug_t), axis = 0)
    concat_aug = concat_aug[:,~np.all(concat_aug == 0, axis = 0)]
    aug_s_trimmed = concat_aug[:len(aug_s)]
    aug_t_trimmed = concat_aug[len(aug_s):]
    return aug_s_trimmed, aug_t_trimmed

def augment_attributes(Gs, Gt, attr_s, attr_t, num_attr, version = "katz", khop = 1, penalty = 0.1, normalize = True): 
    Gs_nodes = list(Gs.nodes())
    Gt_nodes = list(Gt.nodes())
    print(f"This is {version} binning version ")
    if version == "katz":   
        attdict_s = nx.katz_centrality_numpy(Gs,
                                              alpha = 0.01,
                                              beta = 1, 
                                              normalized = False)
        attdict_t = nx.katz_centrality_numpy(Gt,
                                              alpha = 0.01,
                                              beta = 1,
                                              normalized = False)
        if min(attdict_s.values())<0:
            print("arnitiko")
            version="khop"
        elif min(attdict_t.values())<0:
            version="khop"
            print("arnitiko")
    elif version == "eigenvector":

        attdict_s = nx.eigenvector_centrality(Gs, max_iter = 500, tol = 1e-8)
        attdict_t = nx.eigenvector_centrality(Gt, max_iter = 500, tol = 1e-8)
    
    elif version == "pagerank":
        attdict_s = nx.pagerank(Gs,
                                alpha = 0.85,
                                max_iter = 100)
        attdict_t = nx.pagerank(Gt,
                                alpha = 0.85,
                                max_iter = 100)
    
    elif version == "betweenness":
        attdict_s = nx.betweenness_centrality(Gs)
        attdict_t = nx.betweenness_centrality(Gt)
    
    elif version == "closeness":
        attdict_s = nx.closeness_centrality(Gs)
        attdict_t = nx.closeness_centrality(Gt)

        
    if version == "khop":
        
        attdict_s = {key : len(nx.single_source_shortest_path_length
                            (Gs, source = key, cutoff=1))                     
                     for key in Gs_nodes}
        attdict_t = {key : len(nx.single_source_shortest_path_length
                               (Gt, source = key, cutoff=1))
                     for key in Gt_nodes}

        attdict_s_2hop = {key : penalty * len(nx.single_source_shortest_path_length
                            (Gs, source = key, cutoff=2))
                     for key in Gs_nodes}        
        attdict_t_2hop = {key : penalty * len(nx.single_source_shortest_path_length
                               (Gt, source = key, cutoff=2))
                     for key in Gt_nodes}
        dict_seq_2_s = [attdict_s, attdict_s_2hop]
        dict_seq_2_t = [attdict_t, attdict_t_2hop]        
        attdict_s_2hop = reduce(lambda d1,d2: {k: d1.get(k,0)+d2.get(k,0) for k in set(d1)|set(d2)}, dict_seq_2_s)
        attdict_t_2hop = reduce(lambda d1,d2: {k: d1.get(k,0)+d2.get(k,0) for k in set(d1)|set(d2)}, dict_seq_2_t)
                
        attdict_s_3hop = {key : penalty**2 * len(nx.single_source_shortest_path_length
                            (Gs, source = key, cutoff=3))
                  for key in Gs_nodes}
        attdict_t_3hop = {key : penalty**2 * len(nx.single_source_shortest_path_length
                               (Gt, source = key, cutoff=3))
                     for key in Gt_nodes}
        dict_seq_3_s = [attdict_s_2hop, attdict_s_3hop]
        dict_seq_3_t = [attdict_t_2hop, attdict_t_3hop]        
        attdict_s_3hop = reduce(lambda d1,d2: {k: d1.get(k,0)+d2.get(k,0) for k in set(d1)|set(d2)}, dict_seq_3_s)
        attdict_t_3hop = reduce(lambda d1,d2: {k: d1.get(k,0)+d2.get(k,0) for k in set(d1)|set(d2)}, dict_seq_3_t)
        
        if khop == 2:
            attdict_s = attdict_s_2hop
            attdict_t = attdict_t_2hop
            
        elif khop == 3:
            attdict_s = attdict_s_3hop
            attdict_t = attdict_t_3hop
        
        elif khop >= 4:
            print("khop should be set in range [1,3]")
            

    # Consistent normalize
    if normalize == True:        
        attdict_s = {key: (value - min(attdict_s.values())) / (max(attdict_s.values())-min(attdict_s.values()))
                     for key, value in attdict_s.items()}
        attdict_t = {key: (value - min(attdict_t.values())) / (max(attdict_t.values())-min(attdict_t.values()))
                     for key, value in attdict_t.items()}        
        interval = 1 / num_attr
        
    elif normalize == False:
        interval = max(max(attdict_s.values()),max(attdict_t.values())) / num_attr

    init_np_s = np.zeros((Gs.number_of_nodes(), num_attr))
    init_np_t = np.zeros((Gt.number_of_nodes(), num_attr))

    for idx_s, node_s in enumerate(Gs.nodes()):        # assign binning
        cent_node = attdict_s[node_s]
        init_np_s[idx_s, int(cent_node / interval) - 1] = 1

    for idx_t, node_t in enumerate(Gt.nodes()):
        cent_node = attdict_t[node_t]
        init_np_t[idx_t, int(cent_node / interval) - 1] = 1

    new_attr_s = np.append(attr_s, init_np_s, axis = 1)
    new_attr_t = np.append(attr_t, init_np_t, axis = 1)

    if len(attr_s[0]) == 1:
        new_attr_s = new_attr_s[:, 1:]
        new_attr_t = new_attr_t[:, 1:]
        
    return new_attr_s, new_attr_t


def random_flipping_att(att, portion):
    # 몇개 att를 뒤집을 건지
    num_flip = int(0.5 * len(att) * portion)
    att_copy = copy.deepcopy(att)
    for i in range(num_flip):
        # 2개의 인덱스를 (0, len(att)-1)구간 안에서 랜덤하게 뽑은 뒤,
        idx1, idx2 = random.sample(range(0, len(att) - 1), 2)
        # 두 행을 바꿔준다
        att[idx1] = att_copy[idx2]
        att[idx2] = att_copy[idx1]

    return att


def struct_consist_checker(G1, G2, alignment_dict):
    
    G1_nodes = list(G1.nodes()) 
    G2_nodes = list(G2.nodes())    
    # load nodes:deg dict
    degdict_G1 = {key : len(nx.single_source_shortest_path_length(G1, source = key, cutoff=1)) for key in G1_nodes}
    degdict_G2 = {key : len(nx.single_source_shortest_path_length(G2, source = key, cutoff=1)) for key in G2_nodes}
    # compare with alignment_dict
    
    score = 0
    for key in degdict_G1:
        try:            
            if degdict_G1[key] == degdict_G2[alignment_dict[key]]:
                score += 1
        except:
            continue
    str_consist = score/len(alignment_dict)
    print(f"same degree portion is {str_consist:.2f}")
    
    return str_consist
        
def att_consist_checker(G1, G2, attr1, attr2, idx1_dict, idx2_dict, alignment_dict):
    
    G1_nodes = list(G1.nodes()) 
    # load nodes:deg dict

    # compare with alignment_dict
    
    score = 0
    for node in G1_nodes:
        try:            
            if (attr1[idx1_dict[node]] == attr2[idx2_dict[alignment_dict[node]]]).all():
                score += 1
        except:
            continue
    att_consist = score/len(alignment_dict)
    print(f"same attr portion is {att_consist:.2f}")
    
    return att_consist
        


def feat_diff_checker(attr1, attr2):
    print(f"Norm of the vector diff avg is : {np.linalg.norm(np.mean(attr1 - attr2, axis = 0)):.4f}")   
    
def DeleteDuplicatedElement(gt_dict):
    #for am-td
    dellist = [1759,1702,3208,3255,1311,2126,4249,2892,4657,5738,4990,2076]
    #[1786,1725,3231,3275,1349,2148,5641,2915,4661,5605,4991,2077]
    for key in dellist:
        del gt_dict[key]
    return gt_dict    
    