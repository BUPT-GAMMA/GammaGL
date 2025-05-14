import os.path as osp
import scipy.sparse as sp
import numpy as np
import tensorlayerx as tlx
from gammagl.datasets import Planetoid, WikiCS, Coauthor, Amazon
from gammagl.transforms import NormalizeFeatures
from gammagl.data import Dataset
try:
    from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
except ImportError:
    pass  


def get_dataset(dataset_path,name, normalize_features=False, transform=None, if_dpr=True):
    path = dataset_path
    print(path)
    if name in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(path, name)
    elif name in ['arxiv']:
        dataset = PygNodePropPredDataset(name='ogbn-'+name)
    elif name in ['cs', 'physics']:
        dataset = Coauthor(path, name)
    elif name in ['computers', 'photo']:
        dataset = Amazon(path, name)
    elif name in ['wiki']:
        dataset = WikiCS(root='data/wiki')
        dataset.name = 'wiki'
    else:
        raise NotImplementedError

    if transform is not None and normalize_features:
        dataset.transform = lambda x: transform(NormalizeFeatures()(x))
    elif normalize_features:
        dataset.transform = NormalizeFeatures()
    elif transform is not None:
        dataset.transform = transform

    if not if_dpr:
        return dataset

    return Pyg2Dpr(dataset)


class Pyg2Dpr(Dataset):
    def __init__(self, pyg_data, multi_splits=False, **kwargs):
        try:
            splits = pyg_data.get_idx_split()
        except:
            pass

        dataset_name = pyg_data.name
        pyg_data = pyg_data[0]
        n = pyg_data.num_nodes

        edge_index = pyg_data.edge_index
        if hasattr(edge_index, 'device') and str(edge_index.device) != 'cpu':
            try:
                edge_index = tlx.to_device(edge_index, 'cpu')
            except (AttributeError, TypeError):
                pass
        edge_index = tlx.convert_to_numpy(edge_index)
        self.adj = sp.csr_matrix((np.ones(edge_index.shape[1]),
            (edge_index[0], edge_index[1])), shape=(n, n))
            
        x = pyg_data.x
        y = pyg_data.y
        
        if hasattr(x, 'device') and str(x.device) != 'cpu':
            try:
                x = tlx.to_device(x, 'cpu')
            except (AttributeError, TypeError):
                pass
        self.features = tlx.convert_to_numpy(x)
            
        if hasattr(y, 'device') and str(y.device) != 'cpu':
            try:
                y = tlx.to_device(y, 'cpu')
            except (AttributeError, TypeError):
                pass
        self.labels = tlx.convert_to_numpy(y)

        if len(self.labels.shape) == 2 and self.labels.shape[1] == 1:
            self.labels = self.labels.reshape(-1)


def mask_to_index(index, size):
    all_idx = np.arange(size)
    return all_idx[index]


def index_to_mask(index, size):
    mask = tlx.zeros((size, ), dtype=tlx.bool)
    index_tensor = tlx.convert_to_tensor(index)
    ones = tlx.ones_like(index_tensor, dtype=tlx.bool)
    try:
        mask = tlx.scatter_update(mask, index_tensor, ones)
    except (AttributeError, TypeError):
        for i in tlx.convert_to_numpy(index):
            mask = tlx.tensor_scatter_nd_update(mask, [[i]], [True])
    return mask


def weights_init(m):
    if isinstance(m, tlx.nn.Linear):
        weight_attr = 'weights' if hasattr(m, 'weights') else 'weight'
        bias_attr = 'biases' if hasattr(m, 'biases') else 'bias'
        
        if hasattr(m, weight_attr):
            weight = getattr(m, weight_attr)
            tlx.nn.initializers.HeNormal()(weight)
        
        if hasattr(m, bias_attr) and getattr(m, bias_attr) is not None:
            bias = getattr(m, bias_attr)
            tlx.nn.initializers.Zeros()(bias)


def dist_2_label(q_t):
    maxlabel = tlx.reduce_max(q_t, axis=1)
    label = tlx.argmax(q_t, axis=1)
    return maxlabel, label


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape