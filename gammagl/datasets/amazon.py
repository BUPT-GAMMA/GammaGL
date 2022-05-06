from typing import Optional, Callable

import os.path as osp

import numpy as np
from gammagl.data import InMemoryDataset, download_url, Graph
from torch_geometric.io import read_npz
import tensorlayerx as tlx


class Amazon(InMemoryDataset):
    r"""The Amazon Computers and Amazon Photo networks from the
    `"Pitfalls of Graph Neural Network Evaluation"
    <https://arxiv.org/abs/1811.05868>`_ paper.
    Nodes represent goods and edges represent that two goods are frequently
    bought together.
    Given product reviews as bag-of-words node features, the task is to
    map goods to their respective product category.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"Computers"`,
            :obj:`"Photo"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    url = 'https://github.com/shchur/gnn-benchmark/raw/master/data/npz/'

    def __init__(self, root: str, name: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.name = name.lower()
        assert self.name in ['computers', 'photo']
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = np.load(self.processed_paths[0], allow_pickle=True)

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name.capitalize(), 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name.capitalize(), 'processed')

    @property
    def raw_file_names(self) -> str:
        return f'amazon_electronics_{self.name.lower()}.npz'

    @property
    def processed_file_names(self) -> str:
        return tlx.BACKEND+'_data.pt'

    def download(self):
        download_url(self.url + self.raw_file_names, self.raw_dir)

    def process(self):
        data = read_npz(self.raw_paths[0])
        data = data if self.pre_transform is None else self.pre_transform(data)
        self.save_data(self.collate([data]), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}{self.name.capitalize()}()'


import scipy.sparse as sp

# from torch_geometric.utils import remove_self_loops, to_undirected


def read_npz(path):
    with np.load(path) as f:
        return parse_npz(f)


def parse_npz(f):
    x = sp.csr_array((f['attr_data'], f['attr_indices'], f['attr_indptr']),
                      f['attr_shape']).todense()
    x = x.astype(np.float32)
    x[x > 0] = 1

    adj = sp.csr_matrix((f['adj_data'], f['adj_indices'], f['adj_indptr']),
                        f['adj_shape']).tocoo()
    row = adj.row.astype(np.int64)
    col = adj.col.astype(np.int64)
    edge_index = np.vstack((row, col))
    edge_index, _ = remove_self_loops(edge_index)
    edge_index, _ = to_undirected(edge_index, num_nodes=x.size)

    y = f['labels'].astype(np.int64)

    return Graph(x=x, edge_index=edge_index, y=y)


def remove_self_loops(edge_index, edge_attr = None):
    r"""Removes every self-loop in the graph given by :attr:`edge_index`, so
    that :math:`(i,i) \not\in \mathcal{E}` for every :math:`i \in \mathcal{V}`.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """
    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]
    if edge_attr is None:
        return edge_index, None
    else:
        return edge_index, edge_attr[mask]

def to_undirected(
    edge_index,
    edge_attr = None,
    num_nodes = None,
    reduce: str = "add"):
    r"""Converts the graph given by :attr:`edge_index` to an undirected graph
    such that :math:`(j,i) \in \mathcal{E}` for every edge :math:`(i,j) \in
    \mathcal{E}`.
    """
    # Maintain backward compatibility to `to_undirected(edge_index, num_nodes)`
    row, col = edge_index
    row, col = np.hstack((row, col)), np.hstack((col, row))
    edge_index = np.vstack((row, col))

    if edge_attr is not None:
        edge_attr = np.hstack((edge_attr, edge_attr))
    elif edge_attr is not None:
        edge_attr = [np.hstack((e, e)) for e in edge_attr]

    return coalesce(edge_index, edge_attr, num_nodes, num_nodes, reduce)




import numpy as np


def coalesce(index, value, m, n, op="add"):
    "A simplified version of coalesce: Row-wise sorts edge_index and removes its duplicated entries"

    row=index[0]
    col=index[1]
    sparse_sizes=(m, n)

########
### First:Row-wise sorts
########
    idx = np.zeros(col.size + 1)
    idx[1:] = row
    idx[1:] *= sparse_sizes[1]
    idx[1:] += col
    if (idx[1:] < idx[:-1]).any():
        perm = idx[1:].argsort()
        row = row[perm]
        col = col[perm]
        if value is not None:
            value = value[perm]
        idx[1:] = idx[1:][perm]

########
### Second:check if there are repeat index and remove duplicated entries
########
    mask = idx[1:] > idx[:-1]

    if not mask.all():  # Skip if indices are already coalesced.
        row = row[mask]
        col = col[mask]

        # if value is not None:
        #     ptr = mask.nonzero().flatten()
        #     ptr = np.concatenate([ptr, np.full_like(ptr, value.size(0))])
        #     value = segment_csr(value, ptr, reduce=reduce)
    return  np.vstack((row, col)), value


