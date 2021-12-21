import numpy as np
import scipy.sparse as sp
import tensorlayer as tl
from gammagl.data import Graph
from tensorlayer.dataflow import Dataset

def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

# def encode_onehot(labels):
#     classes = set(labels)
#     classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
#     labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
#     return labels_onehot

def encode_labels(labels):
    classes = set(labels)
    classes_dict = {c:i for i,c in enumerate(classes)}
    labels = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    # labels = np.identity(len(classes))[labels] # if onehot needs
    return labels # , len(classes)

def load_data(path):
    idx_features_labels = np.genfromtxt("{}cora.content".format(path), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # labels = tf.convert_to_tensor(encode_onehot(idx_features_labels[:, -1]), dtype=tf.float32)
    labels = tl.convert_to_tensor(encode_labels(idx_features_labels[:, -1]), dtype=tl.int32)
    features = tl.convert_to_tensor(np.array(normalize(features).todense()), dtype=tl.float32)

    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}cora.cites".format(path), dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape).T
    # edges = np.hstack((edges, edges[[1,0], :])) # NOT REASONABLE, some papers cite each other

    # build symmetric adjacency matrix
    adj = sp.coo_matrix((np.ones(edges.shape[1]), (edges[0, :], edges[1, :])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    adj = (adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)).tocoo()
    edges = np.array([adj.col, adj.row]).astype(np.int32)


    idx_train = tl.convert_to_tensor(np.arange(140), dtype=tl.int32)
    idx_val = tl.convert_to_tensor(np.arange(200, 500), dtype=tl.int32)
    idx_test = tl.convert_to_tensor(np.arange(500, 1500), dtype=tl.int32)

    return edges, features, labels, idx_train, idx_val, idx_test

class Cora(Dataset):
    r"""
    A citation network of scientific publications with binary word features.

    Statistics:
        - #Node: 2,708
        - #Edge: 5,429
        - #Class: 7

    Parameters:
        path (str): path to store the dataset
    """
    def __init__(self, path):
        super(Cora,self).__init__()
        self.path = path
        self.num_class = None
        self.feature_dim = None

        self._process()

    def process(self):
        r"""
        Load and preprocess from the raw file of CoRA dataset.

        Returns:
            tuple(graph, idx_train, idx_val, idx_test)
        """
        edges, features, labels, idx_train, idx_val, idx_test = load_data(self.path)
        self.num_class = len(set(labels.numpy()))
        self.feature_dim = features.shape[1]
        graph = Graph(edges, node_feat=features, node_label=labels, num_nodes=labels.shape[0])

        return graph, idx_train, idx_val, idx_test
    
    def _process(self):
        r"""
        Load and preprocess from the raw file of CoRA dataset.

        Returns:
            tuple(graph, idx_train, idx_val, idx_test)
        """
        self.edges, self.features, self.labels, self.idx_train, self.idx_val, self.idx_test = load_data(self.path)
        self.num_class = len(set(self.labels.numpy()))
        self.feature_dim = self.features.shape[1]
        # self.graph = Graph(edges, node_feat=features, node_label=labels, num_nodes=labels.shape[0])


    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.labels[idx]
        return x, y

    def __len__(self):
        return len(self.labels)


