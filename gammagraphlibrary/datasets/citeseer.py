import os
import sys

sys.path.insert(0, os.path.abspath("../../"))
import numpy as np
import scipy.sparse as sp
# import tensorflow as tf
import tensorlayer as tl
from gammagraphlibrary.data import Graph
from .dataset import Dataset

def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def encode_labels(labels):
    classes = set(labels)
    classes_dict = {c:i for i,c in enumerate(classes)}
    labels = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    # labels = np.identity(len(classes))[labels] # if onehot needs
    return labels # , len(classes)

def load_data(path):
    idx_features_labels = np.genfromtxt("{}citeseer.content".format(path), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    idx = np.array(idx_features_labels[:, 0], dtype=object)
    num_nodes = len(idx)
    labels = tl.convert_to_tensor(encode_labels(idx_features_labels[:, -1]), dtype=tl.int32)
    features = tl.convert_to_tensor(np.array(normalize(features).todense()), dtype=tl.float32)

    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}citeseer.cites".format(path), dtype=np.dtype(str))
    edges = []
    for e in edges_unordered:
        e0 = idx_map.get(e[0])
        e1 = idx_map.get(e[1])
        if e0 is not None and e1 is not None:
            edges.append(e0)
            edges.append(e1)
    edges = np.array(edges).reshape((-1,2))
    # edges = np.hstack((edges, edges[[1,0], :])) # NOT REASONABLE, some papers cite each other

    # build symmetric adjacency matrix
    adj = sp.coo_matrix((np.ones(edges.shape[1]), (edges[0, :], edges[1, :])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    adj = (adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)).tocoo()
    edges = np.array([adj.col, adj.row]).astype(np.int32)


    idx_train = tl.convert_to_tensor(np.arange(120), dtype=tl.int32)
    idx_val = tl.convert_to_tensor(np.arange(200, 500), dtype=tl.int32)
    idx_test = tl.convert_to_tensor(np.arange(500, 1500), dtype=tl.int32)

    return edges, features, labels, idx_train, idx_val, idx_test, num_nodes



class CiteSeer(Dataset):
    r"""
    CiteSeer dataset 

    Statistics:
        - #Node: 3,327
        - #Edge: 4,732
        - #Class: 6

    Parameters:
        path (str): path to store the dataset
    """
    def __init__(self, path):
        self.path = path
        self.num_class = None
        self.feature_dim = None

    def process(self):
        r"""
        Load and preprocess from the raw file of CoRA dataset.

        Returns:
            tuple(graph, idx_train, idx_val, idx_test)
        """
        edges, features, labels, idx_train, idx_val, idx_test, num_nodes = load_data(self.path)
        self.num_class = len(set(labels.numpy()))
        self.feature_dim = features.shape[1]
        graph = Graph(edges, num_nodes=num_nodes, node_feat=features, node_label=labels)

        return graph, idx_train, idx_val, idx_test
