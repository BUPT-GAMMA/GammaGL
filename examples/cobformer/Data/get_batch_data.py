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

    def partition_patch(self, n_patches):
        node_mask = metis_partition(g=self.graph, n_patches=n_patches)
        patch = patch2batch(self.graph, node_mask)
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


def get_data_batch(path, name, batch_size=100000):
    if name in ('ogbn-products'):
        dataset = load_ogb_dataset(path, name, batch_size)

    return dataset


def load_ogb_dataset(data_dir, name, batch_size):
    dataset = NCDataset(name)
    ogb_dataset = NodePropPredDataset(name=name, root=f'{data_dir}/ogb')
    graph = ogb_dataset.graph
    graph['edge_index'] = torch.as_tensor(graph['edge_index'])
    graph['node_feat'] = torch.as_tensor(graph['node_feat'])

    label = torch.as_tensor(ogb_dataset.labels).squeeze(-1)

    split_idx = ogb_dataset.get_idx_split()
    train_mask = torch.zeros_like(label, dtype=torch.bool)
    train_mask[split_idx['train']] = True
    valid_mask = torch.zeros_like(label, dtype=torch.bool)
    valid_mask[split_idx['valid']] = True
    test_mask = torch.zeros_like(label, dtype=torch.bool)
    test_mask[split_idx['test']] = True

    graph['edge_index'] = to_undirected(graph['edge_index'])

    return dataset