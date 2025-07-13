import torch
import torch.nn.functional as F
import numpy as np
import torch_geometric
import networkx as nx
import metis
from torch_geometric.datasets import Planetoid
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils import add_remaining_self_loops, to_undirected, remove_self_loops, add_self_loops
from torch_geometric.utils import scatter
import torch_geometric.transforms as T
from infomap import Infomap
from Data.data_utils import *
from ogb.nodeproppred import NodePropPredDataset
import scipy
import scipy.io
import scipy.sparse as sp
import os


class NCDataset(object):
    def __init__(self, name):
        """
        based off of ogb NodePropPredDataset
        https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/dataset.py
        Gives torch tensors instead of numpy arrays
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
            - meta_dict: dictionary that stores all the meta-information about data. Default is None,
                    but when something is passed, it uses its information. Useful for debugging for external contributers.

        Usage after construction:

        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph, label = dataset[0]

        Where the graph is a dictionary of the following form:
        dataset.graph = {'edge_index': edge_index,
                         'edge_feat': None,
                         'node_feat': node_feat,
                         'num_nodes': num_nodes}
        For additional documentation, see OGB Library-Agnostic Loader https://ogb.stanford.edu/docs/nodeprop/

        """

        self.name = name  # original name, e.g., ogbn-proteins
        self.graph = {}
        self.label = None

    def get_idx_split(self, split_type='random', train_prop=.5, valid_prop=.25, label_num_per_class=20):
        """
        split_type: 'random' for random splitting, 'class' for splitting with equal node num per class
        train_prop: The proportion of dataset for train split. Between 0 and 1.
        valid_prop: The proportion of dataset for validation split. Between 0 and 1.
        label_num_per_class: num of nodes per class
        """

        if split_type == 'random':
            ignore_negative = False if self.name == 'ogbn-proteins' else True
            train_mask, valid_mask, test_mask = rand_train_test_idx(
                self.label, train_prop=train_prop, valid_prop=valid_prop, ignore_negative=ignore_negative)
            train_mask = F.pad(train_mask, [0, 1])
            valid_mask = F.pad(valid_mask, [0, 1])
            test_mask = F.pad(test_mask, [0, 1])
            split_idx = {'train': train_mask,
                         'valid': valid_mask,
                         'test': test_mask}
        elif split_type == 'class':
            train_mask, valid_mask, test_mask = class_rand_splits(self.label, label_num_per_class=label_num_per_class)
            train_mask = F.pad(train_mask, [0, 1])
            valid_mask = F.pad(valid_mask, [0, 1])
            test_mask = F.pad(test_mask, [0, 1])
            split_idx = {'train': train_mask,
                         'valid': valid_mask,
                         'test': test_mask}
        return split_idx

    def partition_patch(self, n_patches, load_path=None):
        if load_path is not None:
            patch = torch.load(load_path)
        else:
            if n_patches == 1:
                patch = torch.tensor(range(self.graph['num_nodes'] + 1)).unsqueeze(dim=0)
            else:
                patch = metis_partition(g=self.graph, n_patches=n_patches)
            print('metis done!!!')
        print('patch done!!!')
        self.graph['num_nodes'] += 1
        self.graph['node_feat'] = F.pad(self.graph['node_feat'], [0, 0, 0, 1])
        self.label = F.pad(self.label, [0, 1])
        return patch

    def __getitem__(self, idx):
        assert idx == 0, 'This dataset has only one graph'
        return self.graph, self.label

    def __len__(self):
        return 1

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, len(self))


def get_data(path, name):
    if name in ('Cora', 'CiteSeer', 'PubMed'):
        dataset = load_planetoid_dataset(path, name)
    elif name in ('ogbn-arxiv', 'ogbn-products'):
        dataset = load_ogb_dataset(path, name)
    elif name in ('film'):
        dataset = load_geom_gcn_dataset(path, name)
    elif name in ('deezer'):
        dataset = load_deezer_dataset(path)

    return dataset


def load_planetoid_dataset(data_dir, name):
    # transform = T.NormalizeFeatures()
    torch_dataset = Planetoid(root=data_dir,
                              name=name)
    data = torch_dataset[0]

    edge_index = data.edge_index
    node_feat = data.x
    label = data.y
    num_nodes = data.num_nodes

    dataset = NCDataset(name)

    dataset.train_mask = data.train_mask
    dataset.valid_mask = data.val_mask
    dataset.test_mask = data.test_mask

    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}
    dataset.label = label

    return dataset


def load_ogb_dataset(data_dir, name):
    dataset = NCDataset(name)
    ogb_dataset = NodePropPredDataset(name=name, root=f'{data_dir}/ogb')
    dataset.graph = ogb_dataset.graph
    dataset.graph['edge_index'] = torch.as_tensor(dataset.graph['edge_index'])
    dataset.graph['node_feat'] = torch.as_tensor(dataset.graph['node_feat'])

    dataset.label = torch.as_tensor(ogb_dataset.labels).squeeze(-1)

    split_idx = ogb_dataset.get_idx_split()
    dataset.train_mask = torch.zeros_like(dataset.label, dtype=torch.bool)
    dataset.train_mask[split_idx['train']] = True
    dataset.valid_mask = torch.zeros_like(dataset.label, dtype=torch.bool)
    dataset.valid_mask[split_idx['valid']] = True
    dataset.test_mask = torch.zeros_like(dataset.label, dtype=torch.bool)
    dataset.test_mask[split_idx['test']] = True
    if name == "ogbn-arxiv":
        dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])

    return dataset


def load_geom_gcn_dataset(data_dir, name):
    graph_adjacency_list_file_path = os.path.join(data_dir, 'out1_graph_edges.txt'.format(name))
    graph_node_features_and_labels_file_path = os.path.join(data_dir, 'out1_node_feature_label.txt'.format(name))

    G = nx.DiGraph()
    graph_node_features_dict = {}
    graph_labels_dict = {}

    if name == 'film':
        with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
            graph_node_features_and_labels_file.readline()
            for line in graph_node_features_and_labels_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 3)
                assert (int(line[0]) not in graph_node_features_dict and int(
                    line[0]) not in graph_labels_dict)
                feature_blank = np.zeros(932, dtype=np.uint8)
                feature_blank[np.array(
                    line[1].split(','), dtype=np.uint16)] = 1
                graph_node_features_dict[int(line[0])] = feature_blank
                graph_labels_dict[int(line[0])] = int(line[2])
    else:
        with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
            graph_node_features_and_labels_file.readline()
            for line in graph_node_features_and_labels_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 3)
                assert (int(line[0]) not in graph_node_features_dict and int(
                    line[0]) not in graph_labels_dict)
                graph_node_features_dict[int(line[0])] = np.array(
                    line[1].split(','), dtype=np.uint8)
                graph_labels_dict[int(line[0])] = int(line[2])

    with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
        graph_adjacency_list_file.readline()
        for line in graph_adjacency_list_file:
            line = line.rstrip().split('\t')
            assert (len(line) == 2)
            if int(line[0]) not in G:
                G.add_node(int(line[0]), features=graph_node_features_dict[int(line[0])],
                           label=graph_labels_dict[int(line[0])])
            if int(line[1]) not in G:
                G.add_node(int(line[1]), features=graph_node_features_dict[int(line[1])],
                           label=graph_labels_dict[int(line[1])])
            G.add_edge(int(line[0]), int(line[1]))

    adj = nx.adjacency_matrix(G, sorted(G.nodes()))
    adj = sp.coo_matrix(adj)
    adj = adj + sp.eye(adj.shape[0])
    adj = adj.tocoo().astype(np.float32)
    features = np.array(
        [features for _, features in sorted(G.nodes(data='features'), key=lambda x: x[0])])
    labels = np.array(
        [label for _, label in sorted(G.nodes(data='label'), key=lambda x: x[0])])
    print(features.shape)

    def preprocess_features(feat):
        """Row-normalize feature matrix and convert to tuple representation"""
        rowsum = np.array(feat.sum(1))
        rowsum = (rowsum == 0) * 1 + rowsum
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        feat = r_mat_inv.dot(feat)
        return feat

    features = preprocess_features(features)

    edge_index = torch.from_numpy(
        np.vstack((adj.row, adj.col)).astype(np.int64))
    node_feat = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)
    num_nodes = node_feat.shape[0]
    print(f"Num nodes: {num_nodes}")

    dataset = NCDataset(name)

    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}

    dataset.label = labels
    dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])
    return dataset


def load_deezer_dataset(path):
    filename = 'deezer-europe'
    dataset = NCDataset(filename)
    deezer = scipy.io.loadmat(f'{path}/deezer-europe.mat')

    A, label, features = deezer['A'], deezer['label'], deezer['features']
    edge_index = torch.tensor(A.nonzero(), dtype=torch.long)
    node_feat = torch.tensor(features.todense(), dtype=torch.float)
    label = torch.tensor(label, dtype=torch.long).squeeze()
    num_nodes = label.shape[0]

    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': node_feat,
                     'num_nodes': num_nodes}
    # dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])
    dataset.label = label
    return dataset
