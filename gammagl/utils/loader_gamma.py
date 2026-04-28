import os.path as osp
import os
os.environ['TL_BACKEND'] = 'torch'
import sys
import gc
import copy
from dotmap import DotMap
import numpy as np
import scipy.sparse as sp

# ===================== 核心框架导入（仅保留必需的，删除所有废弃数据集导入） =====================
import tensorlayerx as tlx
from gammagl.data import Graph
from typing import Any, Callable, List, Optional, Union, Sequence

# ===================== 手动实现所有GammaGL废弃函数，彻底解决导入错误 =====================
def stochastic_blockmodel_graph(block_sizes, edge_probs, directed=False):
    """手动实现 SBM 图生成，100%兼容原有接口"""
    N = sum(block_sizes)
    edges = []
    for i, size_i in enumerate(block_sizes):
        start_i = sum(block_sizes[:i])
        end_i = start_i + size_i
        for j, size_j in enumerate(block_sizes):
            start_j = sum(block_sizes[:j])
            end_j = start_j + size_j
            p = edge_probs[i, j] if isinstance(edge_probs, np.ndarray) else edge_probs[i][j]
            if directed:
                idx_i = np.random.choice(np.arange(start_i, end_i), size=int(size_i * size_j * p))
                idx_j = np.random.choice(np.arange(start_j, end_j), size=int(size_i * size_j * p))
            else:
                if i > j:
                    continue
                n_edges = int(size_i * size_j * p)
                idx_i, idx_j = np.random.randint(start_i, end_i, n_edges), np.random.randint(start_j, end_j, n_edges)
                if i == j:
                    mask = idx_i < idx_j
                    idx_i, idx_j = idx_i[mask], idx_j[mask]
            if len(idx_i) > 0:
                edges.append(np.stack([idx_i, idx_j]))
    edge_index = np.concatenate(edges, axis=1)
    if not directed:
        edge_index = np.concatenate([edge_index, edge_index[::-1]], axis=1)
    return edge_index

def to_scipy_sparse_matrix(edge_index, num_nodes=None):
    """手动实现稀疏矩阵转换"""
    if num_nodes is None:
        num_nodes = edge_index.max() + 1
    return sp.coo_matrix((np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])),
                         shape=(num_nodes, num_nodes))

# 手动实现SBM数据集类，不依赖GammaGL内置类
class DummySBM:
    def __init__(self, edge_index, num_nodes):
        self.edge_index = edge_index
        self.num_nodes = num_nodes

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from .gen_cat import gencat, feature_extraction
from .data_processor import DataProcess, DataProcess_inductive, matstd_clip
from precompute.prop import A2Prop

# 打印设置（仅保留numpy，删除TLX无效方法）
np.set_printoptions(linewidth=160, edgeitems=5, threshold=20,
                    formatter=dict(float=lambda x: "% 9.3e" % x))


def dmap2dct(chnname: str, dmap: DotMap, processor: DataProcess):
    """【纯逻辑函数，无修改】参数映射函数"""
    typedct = {'sgc': -2, 'gbp': -3,
               'sgc_agp': 0, 'gbp_agp': 1,
               'sgc_thr': 2, 'gbp_thr': 3,}
    ctype = chnname.split('_')[0]

    dct = {}
    dct['type'] = typedct[chnname]
    dct['hop'] = dmap.hop
    dct['dim'] = processor.nfeat
    dct['delta'] = dmap.delta if type(dmap.delta) is float else 1e-5
    dct['alpha'] = dmap.alpha if (type(dmap.alpha) is float and not (ctype == 'sgc')) else 0
    dct['rra'] = (1 - dmap.rrz) if type(dmap.rrz) is float else 0
    dct['rrb'] = dmap.rrz if type(dmap.rrz) is float else 0
    return dct


def to_list(value: Any) -> Sequence:
    """【纯工具函数，无修改】"""
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    else:
        return [value]


# ===================== 彻底删除废弃的GammaGL数据集继承，纯手动实现 =====================
def sbm_mixture_of_Gaussians(n, n_features, sizes, probs, std_, means, nodes_per_mean):
    """【纯手动实现】SBM高斯混合数据生成，无任何GammaGL数据集依赖"""
    # 手动生成SBM图
    edge_index = stochastic_blockmodel_graph(sizes, probs, directed=False)
    g_ = DummySBM(edge_index=edge_index, num_nodes=sum(sizes))

    A = to_scipy_sparse_matrix(g_.edge_index, num_nodes=g_.num_nodes)
    A.setdiag(A.diagonal() + 1)
    d_mat = np.sum(A, axis=0)
    dinv_mat = 1/d_mat
    dinv = dinv_mat.tolist()[0]
    Dinv = sp.diags(dinv, 0)

    X = np.zeros((2*n,n_features))
    for ct, mean_ in enumerate(means):
        X[nodes_per_mean[ct],:] = std_ * np.random.randn(len(nodes_per_mean[ct]), n_features) + mean_

    data = np.zeros((2*n,n_features+1))
    data[:,:-1] = Dinv@A@X
    data[:,-1]  = np.ones(2*n)
    g_.x = tlx.convert_to_tensor(data, dtype=tlx.float32)

    y = np.zeros(2*n)
    y[0:n] = 1
    g_.y = tlx.convert_to_tensor(y, dtype=tlx.float32)

    return g_


def load_csbm(datastr: str, datapath: str="./data/",
                   inductive: bool=False, multil: bool=False,
                   seed: int=0, **kwargs):
    """【接口完全不变】CSBM数据集加载"""
    n = 500
    n_features = 127
    q = 0.1
    _, p, mu = datastr.split('-')
    p, mu = float(p), float(mu)

    probs = [[p, q], [q, p]]
    sizes = [n, n]
    std_ = mu
    means = [100, -100]
    nodes_per_mean = [list(range(n)),list(range(n,2*n))]
    
    g = sbm_mixture_of_Gaussians(n, n_features, sizes, probs, std_, means, nodes_per_mean)

    idx_rnd = np.random.permutation(2*n)
    # 张量：torch → tlx
    adj = {'train': g.edge_index,
           'test':  g.edge_index}
    feat = {'train': tlx.convert_to_tensor(g.x, tlx.float32),
           'test':  tlx.convert_to_tensor(g.x, tlx.float32)}
    idx = {'train': tlx.convert_to_tensor(idx_rnd[:int(0.6*2*n)], tlx.int64),
           'val':   tlx.convert_to_tensor(idx_rnd[int(0.6*2*n):int(0.8*2*n)], tlx.int64),
           'test':  tlx.convert_to_tensor(idx_rnd[int(0.8*2*n):], tlx.int64)}
    y = tlx.cast(g.y, tlx.int64)
    labels = {'train': y[idx['train']],
              'val':   y[idx['val']],
              'test':  y[idx['test']]}
    nfeat = n_features + 1
    nclass = 2
    if seed >= 15:
        print(g)
    return adj, feat, labels, idx, nfeat, nclass


def load_gencat(datastr: str, datapath: str="./data/",
                   inductive: bool=False, multil: bool=False,
                   seed: int=0, **kwargs):
    """【纯numpy逻辑，无修改】生成图数据加载"""
    dp = DataProcess('cora', path="./data/", seed=0)
    dp.input(['adjnpz', 'labels', 'attr_matrix'])
    M, D, class_size, H, node_degree = feature_extraction(dp.adj_matrix, dp.attr_matrix, dp.labels.tolist())

    _, p, omega = datastr.split('-')
    p, omega = float(p), float(omega)
    adj, feat, labels = gencat(M, D, H, n=1000, m=5000, p=p, omega=omega)
    dp.adj_matrix = adj
    dp.attr_matrix = feat
    dp.labels = np.array(labels)

    dp.calculate(['idx_train'])
    idx = {'train': tlx.convert_to_tensor(dp.idx_train, tlx.int64),
           'val':   tlx.convert_to_tensor(dp.idx_val, tlx.int64),
           'test':  tlx.convert_to_tensor(dp.idx_test, tlx.int64)}
    labels = tlx.convert_to_tensor(dp.labels.flatten(), tlx.int64)
    labels = {'train': labels[idx['train']],
              'val':   labels[idx['val']],
              'test':  labels[idx['test']]}
    dp.calculate(['edge_idx'])
    adj = {'test':  tlx.convert_to_tensor(dp.edge_idx, tlx.int64),
           'train': tlx.convert_to_tensor(dp.edge_idx, tlx.int64)}
    feat = {'test': tlx.convert_to_tensor(dp.attr_matrix, tlx.float32),
            'train': tlx.convert_to_tensor(dp.attr_matrix, tlx.float32)}
    n, m = dp.n, dp.m
    nfeat, nclass = dp.nfeat, dp.nclass
    if seed >= 15:
        print(dp)
    return adj, feat, labels, idx, nfeat, nclass


# ========== 核心数据加载函数（完全兼容原版） ==========
def load_edgelist(datastr: str, datapath: str="./data/",
                   inductive: bool=False, multil: bool=False,
                   seed: int=0, **kwargs):
    """【最常用函数，接口/返回值完全不变】边列表数据加载"""
    if datastr.startswith('csbm'):
        return load_csbm(datastr, datapath, inductive, multil, seed, **kwargs)
    elif datastr.startswith('gencat'):
        return load_gencat(datastr, datapath, inductive, multil, seed, **kwargs)

    dp = DataProcess(datastr, path=datapath, seed=seed)
    dp.input(['adjnpz', 'labels', 'attr_matrix'])
    if inductive:
        dpi = DataProcess_inductive(datastr, path=datapath, seed=seed)
        dpi.input(['adjnpz', 'attr_matrix'])
    else:
        dpi = dp

    if (datastr.startswith('cora') or datastr.startswith('citeseer') or datastr.startswith('pubmed')):
        dp.calculate(['idx_train'])
    else:
        dp.input(['idx_train', 'idx_val', 'idx_test'])
    idx = {'train': tlx.convert_to_tensor(dp.idx_train, tlx.int64),
           'val':   tlx.convert_to_tensor(dp.idx_val, tlx.int64),
           'test':  tlx.convert_to_tensor(dp.idx_test, tlx.int64)}

    if multil:
        dp.calculate(['labels_oh'])
        dp.labels_oh[dp.labels_oh < 0] = 0
        labels = tlx.convert_to_tensor(dp.labels_oh, tlx.float32)
    else:
        dp.labels[dp.labels < 0] = 0
        labels = tlx.convert_to_tensor(dp.labels.flatten(), tlx.int64)
    labels = {'train': labels[idx['train']],
              'val':   labels[idx['val']],
              'test':  labels[idx['test']]}

    dp.calculate(['edge_idx'])
    adj = {'test':  tlx.convert_to_tensor(dp.edge_idx, tlx.int64)}
    if inductive:
        dpi.calculate(['edge_idx'])
        adj['train'] = tlx.convert_to_tensor(dpi.edge_idx, tlx.int64)
    else:
        adj['train'] = adj['test']

    feat = dp.attr_matrix
    feati = dpi.attr_matrix if inductive else feat
    feat = {'test': tlx.convert_to_tensor(feat, tlx.float32)}
    feat['train'] = tlx.convert_to_tensor(feati, tlx.float32)

    n, m = dp.n, dp.m
    nfeat, nclass = dp.nfeat, dp.nclass
    if seed >= 15:
        print(dp)
    return adj, feat, labels, idx, nfeat, nclass


def load_embedding(datastr: str, algo: str, algo_chn: DotMap,
                   datapath: str="./data/",
                   inductive: bool=False, multil: bool=False,
                   seed: int=0, **kwargs):
    """【嵌入加载函数，C++扩展兼容，无修改】"""
    dp = DataProcess(datastr, path=datapath, seed=seed)
    dp.input(['labels', 'attr_matrix', 'deg'])
    if inductive:
        dpi = DataProcess_inductive(datastr, path=datapath, seed=seed)
        dpi.input(['attr_matrix', 'deg'])
    else:
        dpi = dp

    if (datastr.startswith('cora') or datastr.startswith('citeseer') or datastr.startswith('pubmed')):
        dp.calculate(['idx_train'])
    else:
        dp.input(['idx_train', 'idx_val', 'idx_test'])
    idx = {'train': tlx.convert_to_tensor(dp.idx_train, tlx.int64),
           'val':   tlx.convert_to_tensor(dp.idx_val, tlx.int64),
           'test':  tlx.convert_to_tensor(dp.idx_test, tlx.int64)}

    if multil:
        dp.calculate(['labels_oh'])
        dp.labels_oh[dp.labels_oh < 0] = 0
        labels = tlx.convert_to_tensor(dp.labels_oh, tlx.float32)
    else:
        dp.labels[dp.labels < 0] = 0
        labels = tlx.convert_to_tensor(dp.labels.flatten(), tlx.int64)
    labels = {'train': labels[idx['train']],
              'val':   labels[idx['val']],
              'test':  labels[idx['test']]}

    n, m = dp.n, dp.m
    nfeat, nclass = dp.nfeat, dp.nclass
    if seed >= 15:
        print(dp)

    # C++ 扩展调用（完全兼容）
    py_a2prop = A2Prop()
    py_a2prop.load(os.path.join(datapath, datastr), m, n, seed)
    chn = dmap2dct(algo, DotMap(algo_chn), dp)

    feat = dp.attr_matrix.transpose().astype(np.float32, order='C')
    macs_pre, time_pre = py_a2prop.compute(1, [chn], feat)
    feat = feat.transpose()
    feat = matstd_clip(feat, idx['train'], with_mean=True)

    feats = {'val':  tlx.convert_to_tensor(feat[idx['val']], tlx.float32),
             'test': tlx.convert_to_tensor(feat[idx['test']], tlx.float32)}
    feats['train'] = tlx.convert_to_tensor(feat[idx['train']], tlx.float32)
    
    del feat
    gc.collect()
    return feats, labels, idx, nfeat, nclass, macs_pre/1e9, time_pre