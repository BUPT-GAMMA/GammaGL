import os.path as osp
import scipy.sparse as sp
import numpy as np
import tensorlayerx as tlx

from gammagl.datasets import Planetoid, WikiCS, Coauthor, Amazon
from gammagl.transforms import NormalizeFeatures
from gammagl.data import Dataset

# OGB数据集的导入
try:
    from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
except ImportError:
    pass  # 如果不需要OGB数据集，可以忽略这个导入错误


def get_dataset(name, normalize_features=False, transform=None, if_dpr=True):
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', name)
    if name in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(path, name)
    # elif name in ['corafull']:
    #     dataset = CoraFull(path)
    elif name in ['arxiv']:
        # 需要确认GammaGL是否支持OGB数据集
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
        # 使用GammaGL的转换函数
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

        # 确保 edge_index 在 CPU 上，然后再转换为 NumPy 数组
        edge_index = pyg_data.edge_index
        if hasattr(edge_index, 'device') and str(edge_index.device) != 'cpu':
            try:
                # 尝试使用to_device，如果不可用则使用其他方法
                edge_index = tlx.to_device(edge_index, 'cpu')
            except (AttributeError, TypeError):
                # 如果to_device不可用，尝试使用convert_to_numpy直接转换
                pass
        edge_index = tlx.convert_to_numpy(edge_index)
            
        self.adj = sp.csr_matrix((np.ones(edge_index.shape[1]),
            (edge_index[0], edge_index[1])), shape=(n, n))
            
        # 确保特征和标签在 CPU 上，然后再转换为 NumPy 数组
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
    # 使用TensorLayerX创建张量
    mask = tlx.zeros((size, ), dtype=tlx.bool)
    # 确保index是张量
    index_tensor = tlx.convert_to_tensor(index)
    # 创建与index相同形状的全1张量
    ones = tlx.ones_like(index_tensor, dtype=tlx.bool)
    # 使用scatter_update更新mask
    try:
        mask = tlx.scatter_update(mask, index_tensor, ones)
    except (AttributeError, TypeError):
        # 如果scatter_update不可用，使用循环实现
        for i in tlx.convert_to_numpy(index):
            mask = tlx.tensor_scatter_nd_update(mask, [[i]], [True])
    return mask


def weights_init(m):
    # 根据TensorLayerX的模型初始化方式进行修改
    if isinstance(m, tlx.nn.Linear):
        # 检查权重属性名称
        weight_attr = 'weights' if hasattr(m, 'weights') else 'weight'
        bias_attr = 'biases' if hasattr(m, 'biases') else 'bias'
        
        # 初始化权重
        if hasattr(m, weight_attr):
            weight = getattr(m, weight_attr)
            tlx.nn.initializers.HeNormal()(weight)
        
        # 初始化偏置
        if hasattr(m, bias_attr) and getattr(m, bias_attr) is not None:
            bias = getattr(m, bias_attr)
            tlx.nn.initializers.Zeros()(bias)


def dist_2_label(q_t):
    # 使用TensorLayerX的操作
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