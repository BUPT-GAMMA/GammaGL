import numpy as np
import tensorlayerx as tlx
import torch
import networkx as nx
from sklearn.decomposition import PCA
from torch_geometric.data import HeteroData
from torch_geometric.utils import from_networkx
from torch_geometric.datasets import GitHub as PyGGitHub
from gammagl.data import Graph
import scipy.sparse as sp
import pickle


class KarateClubDataset:
    def __init__(self):
        G = nx.karate_club_graph()
        data = from_networkx(G)
        data.num_classes = 4
        data.num_features = 34
        data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.train_mask[:20] = True
        data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.val_mask[20:30] = True
        data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.test_mask[30:] = True
        self.data = data


class GitHubDataset:
    def __init__(self, root='data/github'):
        dataset = PyGGitHub(root=root)
        data = dataset[0]
        data.num_features = data.x.shape[1]
        data.num_classes = int(data.y.max().item()) + 1
        self.data = data

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.data


class AirportsDataset:
    def __init__(self, root='data/airports', name='brazil'):
        with open(f'{root}/{name}-airports.edgelist', 'r') as f:
            edges = f.readlines()
        edges = [[int(x) for x in e.strip().split()] for e in edges]
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        num_nodes = edge_index.max().item() + 1
        x = torch.eye(num_nodes)
        data = HeteroData()
        data['node'].x = x
        data['node', 'to', 'node'].edge_index = edge_index
        data.num_features = num_nodes
        data.num_nodes = num_nodes
        
        with open(f'{root}/{name}-airports.labels', 'r') as f:
            labels = f.readlines()
        labels = [int(x.strip()) for x in labels]
        data.y = torch.tensor(labels[:num_nodes], dtype=torch.long)
        data.num_classes = max(labels) + 1
        
        num_nodes = data.num_nodes
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        indices = torch.randperm(num_nodes)
        train_size = int(0.5 * num_nodes)
        val_size = int(0.25 * num_nodes)
        train_mask[indices[:train_size]] = True
        val_mask[indices[train_size:train_size + val_size]] = True
        test_mask[indices[train_size + val_size:]] = True
        
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
        self.data = data

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.data
