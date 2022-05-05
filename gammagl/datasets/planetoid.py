import itertools
from typing import Callable, List, Optional
from gammagl.data import download_url
from gammagl.data import InMemoryDataset
from gammagl.utils.io import txt_array, read_txt_array

try:
    import cPickle as pickle
except ImportError:
    import pickle


class Planetoid(InMemoryDataset):
    r"""
    The citation network datasets "Cora", "CiteSeer" and "PubMed" from the
    `"Revisiting Semi-Supervised Learning with Graph Embeddings"
    <https://arxiv.org/abs/1603.08861>`_ paper.
    Nodes represent documents and edges represent citation links.
    Training, validation and test splits are given by binary masks.
    Args:
        root (str): Root directory where the dataset should be saved.
        name (str): The name of the dataset (:obj:`"Cora"`,
            :obj:`"CiteSeer"`, :obj:`"PubMed"`).
        split (string): The type of dataset split
            (:obj:`"public"`, :obj:`"full"`, :obj:`"random"`).
            If set to :obj:`"public"`, the split will be the public fixed split
            from the
            `"Revisiting Semi-Supervised Learning with Graph Embeddings"
            <https://arxiv.org/abs/1603.08861>`_ paper.
            If set to :obj:`"full"`, all nodes except those in the validation
            and test sets will be used for training (as in the
            `"FastGCN: Fast Learning with Graph Convolutional Networks via
            Importance Sampling" <https://arxiv.org/abs/1801.10247>`_ paper).
            If set to :obj:`"random"`, train, validation, and test sets will be
            randomly generated, according to :obj:`num_train_per_class`,
            :obj:`num_val` and :obj:`num_test`. (default: :obj:`"public"`)
        num_train_per_class (int, optional): The number of training samples
            per class in case of :obj:`"random"` split. (default: :obj:`20`)
        num_val (int, optional): The number of validation samples in case of
            :obj:`"random"` split. (default: :obj:`500`)
        num_test (int, optional): The number of test samples in case of
            :obj:`"random"` split. (default: :obj:`1000`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    Stats:
        .. list-table::
            :widths: 10 10 10 10 10
            :header-rows: 1
            * - Name
              - #nodes
              - #edges
              - #features
              - #classes
            * - Cora
              - 2,708
              - 10,556
              - 1,433
              - 7
            * - CiteSeer
              - 3,327
              - 9,104
              - 3,703
              - 6
            * - PubMed
              - 19,717
              - 88,648
              - 500
              - 3
    """

    url = 'https://github.com/kimiyoung/planetoid/raw/master/data'

    def __init__(self, root: str, name: str, split: str = "public",
                 num_train_per_class: int = 20, num_val: int = 500,
                 num_test: int = 1000, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.name = name

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = self.load_data(self.processed_paths[0])
        self.split = split
        assert self.split in ['public', 'full', 'random']

        if split == 'full':
            data = self.get(0)
            data.train_mask.fill_(True)
            data.train_mask[data.val_mask | data.test_mask] = False
            self.data, self.slices = self.collate([data])

        elif split == 'random':
            data = self.get(0)
            data.train_mask.fill_(False)
            for c in range(self.num_classes):
                idx = (data.y == c).nonzero(as_tuple=False).view(-1)
                idx = idx[torch.randperm(idx.size(0))[:num_train_per_class]]
                data.train_mask[idx] = True

            remaining = (~data.train_mask).nonzero(as_tuple=False).view(-1)
            remaining = remaining[torch.randperm(remaining.size(0))]

            data.val_mask.fill_(False)
            data.val_mask[remaining[:num_val]] = True

            data.test_mask.fill_(False)
            data.test_mask[remaining[num_val:num_val + num_test]] = True

            self.data, self.slices = self.collate([data])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        names = ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']
        return [f'ind.{self.name.lower()}.{name}' for name in names]

    @property
    def processed_file_names(self) -> str:
        return tlx.BACKEND+'_data.pt'

    def download(self):
        for name in self.raw_file_names:
            download_url(f'{self.url}/{name}', self.raw_dir)

    def process(self):
        data = read_planetoid_data(self.raw_dir, self.name)
        data = data if self.pre_transform is None else self.pre_transform(data)
        self.save_data(self.collate([data]), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name}()'


import os.path as osp
import sys
from itertools import repeat
import tensorlayerx as tlx
import numpy as np
import torch
from gammagl.data import Graph


try:
    import cPickle as pickle
except ImportError:
    import pickle


def read_planetoid_data(folder, prefix):
    names = ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']
    items = [read_file(folder, prefix, name) for name in names]
    x, tx, allx, y, ty, ally, graph, test_index = items
    train_index = np.arange(y.shape[0])
    val_index = np.arange(y.shape[0], y.shape[0] + 500)
    sorted_test_index = np.sort(test_index)

    if prefix.lower() == 'citeseer':
        # There are some isolated nodes in the Citeseer graph, resulting in
        # none consecutive test indices. We need to identify them and add them
        # as zero vectors to `tx` and `ty`.
        len_test_indices = (test_index.max() - test_index.min()).item() + 1

        tx_ext = np.zeros((len_test_indices, tx.shape[1]))
        tx_ext[sorted_test_index - test_index.min(), :] = tx
        ty_ext = np.zeros((len_test_indices, ty.shape[1]))
        ty_ext[sorted_test_index - test_index.min(), :] = ty

        tx, ty = tx_ext, ty_ext

    if prefix.lower() == 'nell.0.001':
        tx_ext = torch.zeros(len(graph) - allx.size(0), x.size(1))
        tx_ext[sorted_test_index - allx.size(0)] = tx

        ty_ext = torch.zeros(len(graph) - ally.size(0), y.size(1))
        ty_ext[sorted_test_index - ally.size(0)] = ty

        tx, ty = tx_ext, ty_ext

        x = torch.cat([allx, tx], dim=0)
        x[test_index] = x[sorted_test_index]

        # Creating feature vectors for relations.
        row, col, value = SparseTensor.from_dense(x).coo()
        rows, cols, values = [row], [col], [value]

        mask1 = index_to_mask(test_index, size=len(graph))
        mask2 = index_to_mask(torch.arange(allx.size(0), len(graph)),
                              size=len(graph))
        mask = ~mask1 | ~mask2
        isolated_index = mask.nonzero(as_tuple=False).view(-1)[allx.size(0):]

        rows += [isolated_index]
        cols += [torch.arange(isolated_index.size(0)) + x.size(1)]
        values += [torch.ones(isolated_index.size(0))]

        x = SparseTensor(row=torch.cat(rows), col=torch.cat(cols),
                         value=torch.cat(values))
    else:
        x = np.concatenate([allx, tx])
        x[test_index] = x[sorted_test_index]

    y = np.concatenate([ally, ty])
    y[test_index] = y[sorted_test_index]

    train_mask = index_to_mask(train_index, size=y.shape[0])
    val_mask = index_to_mask(val_index, size=y.shape[0])
    test_mask = index_to_mask(test_index, size=y.shape[0])

    edge_index = edge_index_from_dict(graph, num_nodes=y.shape[0])

    # remove duplicated edges
    edge_index = list(k for k, _ in itertools.groupby(sorted(edge_index)))
    edge = np.array(edge_index).T
    ind = np.argsort(edge[1], axis=0)
    edge = np.array(edge.T[ind]).T

    data = Graph(edge_index=edge, x=x, y=y)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    return data


def read_file(folder, prefix, name):
    path = osp.join(folder, f'ind.{prefix.lower()}.{name}')

    if name == 'test.index':
        return read_txt_array(path)

    with open(path, 'rb') as f:
        if sys.version_info > (3, 0):
            out = pickle.load(f, encoding='latin1')
        else:
            out = pickle.load(f)

    if name == 'graph':
        return out

    out = out.todense() if hasattr(out, 'todense') else out
    out = np.array(out)
    return out


def edge_index_from_dict(graph_dict, num_nodes=None, coalesce=True):
    # row, col = [], []
    # for key, value in graph_dict.items():
    #     row += repeat(key, len(value))
    #     col += value
    # edge_index = np.stack([np.array(row), np.array(col)], axis=0)
    # 我改了，这样肯定行，老三样数据集基本都这么处理
    edge_index = []
    for src, dst in graph_dict.items():  # 格式为 {index：[index_of_neighbor_nodes]}
        edge_index.extend([src, v] for v in dst) # 起始点， 终点
        edge_index.extend([v, src] for v in dst)
    # if coalesce:
    #     # NOTE: There are some duplicated edges and self loops in the datasets.
    #     #       Other implementations do not remove them!
    #     edge_index, _ = remove_self_loops(edge_index)
    #     edge_index, _ = coalesce_fn(edge_index, None, num_nodes, num_nodes)
    return edge_index


def index_to_mask(index, size):
    mask = np.zeros((size, ), dtype=bool)
    mask[index] = 1
    return mask
