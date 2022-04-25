import os
import os.path as osp
import copy
import shutil
from typing import Callable, List, Optional

try:
    import cPickle as pickle
except ImportError:
    import pickle


from gammagl.data import Graph, InMemoryDataset, download_url, extract_zip


class TUDataset(InMemoryDataset):
    r"""A variety of graph kernel benchmark datasets, *.e.g.* "IMDB-BINARY",
    "REDDIT-BINARY" or "PROTEINS", collected from the `TU Dortmund University
    <https://chrsmrrs.github.io/datasets>`_.
    In addition, this dataset wrapper provides `cleaned dataset versions
    <https://github.com/nd7141/graph_datasets>`_ as motivated by the
    `"Understanding Isomorphism Bias in Graph Data Sets"
    <https://arxiv.org/abs/1910.12091>`_ paper, containing only non-isomorphic
    graphs.
    .. note::
        Some datasets may not come with any node labels.
        You can then either make use of the argument :obj:`use_node_attr`
        to load additional continuous node attributes (if present) or provide
        synthetic node features using transforms such as
        like :class:`torch_geometric.transforms.Constant` or
        :class:`torch_geometric.transforms.OneHotDegree`.
    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The `name
            <https://chrsmrrs.github.io/datasets/docs/datasets/>`_ of the
            dataset.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
        use_node_attr (bool, optional): If :obj:`True`, the dataset will
            contain additional continuous node attributes (if present).
            (default: :obj:`False`)
        use_edge_attr (bool, optional): If :obj:`True`, the dataset will
            contain additional continuous edge attributes (if present).
            (default: :obj:`False`)
        cleaned: (bool, optional): If :obj:`True`, the dataset will
            contain only non-isomorphic graphs. (default: :obj:`False`)
    Stats:
        .. list-table::
            :widths: 20 10 10 10 10 10
            :header-rows: 1
            * - Name
              - #graphs
              - #nodes
              - #edges
              - #features
              - #classes
            * - MUTAG
              - 188
              - ~17.9
              - ~39.6
              - 7
              - 2
            * - ENZYMES
              - 600
              - ~32.6
              - ~124.3
              - 3
              - 6
            * - PROTEINS
              - 1,113
              - ~39.1
              - ~145.6
              - 3
              - 2
            * - COLLAB
              - 5,000
              - ~74.5
              - ~4914.4
              - 0
              - 3
            * - IMDB-BINARY
              - 1,000
              - ~19.8
              - ~193.1
              - 0
              - 2
            * - REDDIT-BINARY
              - 2,000
              - ~429.6
              - ~995.5
              - 0
              - 2
            * - ...
              -
              -
              -
              -
              -
    """

    url = 'https://www.chrsmrrs.com/graphkerneldatasets'
    cleaned_url = ('https://raw.githubusercontent.com/nd7141/'
                   'graph_datasets/master/datasets')

    def __init__(self, root: str, name: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 use_node_attr: bool = False, use_edge_attr: bool = False,
                 cleaned: bool = False):
        self.name = name
        self.cleaned = cleaned
        super().__init__(root, transform, pre_transform, pre_filter)
        with open(self.processed_paths[0], 'rb') as f:
            self.data, self.slices = pickle.load(f)
        # if self.graph.x is not None and not use_node_attr:
        #     num_node_attributes = self.num_node_attributes
        #     self.graph.x = self.graph.x[:, num_node_attributes:]
        # if self.graph.edge_feat is not None and not use_edge_attr:
        #     num_edge_attributes = self.num_edge_attributes
        #     self.graph.edge_feat = self.graph.edge_attr[:, num_edge_attributes:]

    @property
    def raw_dir(self) -> str:
        name = f'raw{"_cleaned" if self.cleaned else ""}'
        return osp.join(self.root, self.name, name)

    @property
    def processed_dir(self) -> str:
        name = f'processed{"_cleaned" if self.cleaned else ""}'
        return osp.join(self.root, self.name, name)

    @property
    def num_node_labels(self) -> int:
        if self.data.x is None:
            return 0
        for i in range(self.data.x.shape[1]):
            x = self.data.x[:, i:]
            if ((x == 0) | (x == 1)).sum() == x.shape[0] and (x.sum(dim=1) == 1).sum() == x.shape[0]:
                return self.data.x.shape[1] - i
        return 0

    @property
    def num_node_attributes(self) -> int:
        if self.data.x is None:
            return 0
        return self.data.x.shape[1] - self.num_node_labels

    @property
    def num_edge_labels(self) -> int:
        if self.data.edge_attr is None:
            return 0
        for i in range(self.data.edge_attr.shape[1]):
            if self.data.edge_attr[:, i:].sum() == self.data.edge_attr.shape[0]:
                return self.data.edge_attr.shape[1] - i
        return 0

    @property
    def num_edge_attributes(self) -> int:
        if self.data.edge_feat is None:
            return 0
        return self.data.edge_feat.shape[1] - self.num_edge_labels

    @property
    def raw_file_names(self) -> List[str]:
        names = ['A', 'graph_indicator']
        return [f'{self.name}_{name}.txt' for name in names]

    @property
    def processed_file_names(self) -> str:
        return 'graph.pt'

    def download(self):
        url = self.cleaned_url if self.cleaned else self.url
        folder = osp.join(self.root, self.name)
        path = download_url(f'{url}/{self.name}.zip', folder)
        extract_zip(path, folder)
        os.unlink(path)
        shutil.rmtree(self.raw_dir)
        os.rename(osp.join(folder, self.name), self.raw_dir)

    def process(self):
        self.data, self.slices = read_tu_data(self.raw_dir, self.name)

        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)
        with open(self.processed_paths[0], 'wb') as f:
            pickle.dump((self.data, self.slices), f)

    def __repr__(self) -> str:
        return f'{self.name}({len(self)})'

    
import glob
import os
import os.path as osp

import numpy as np
import torch
import torch.nn.functional as F
# from torch_sparse import coalesce
#
# from torch_geometric.data import Data
from gammagl.utils.io import read_txt_array
# from torch_geometric.utils import remove_self_loops

names = [
    'A', 'graph_indicator', 'node_labels', 'node_attributes'
                                           'edge_labels', 'edge_attributes', 'graph_labels', 'graph_attributes'
]


def read_tu_data(folder, prefix):
    files = glob.glob(osp.join(folder, f'{prefix}_*.txt'))
    names = [f.split(os.sep)[-1][len(prefix) + 1:-4] for f in files]
    
    edge_index = read_file(folder, prefix, 'A', np.int32).transpose() - 1
    batch = read_file(folder, prefix, 'graph_indicator', np.int32) - 1
    
    node_attributes = node_labels = None
    if 'node_attributes' in names:
        node_attributes = read_file(folder, prefix, 'node_attributes', dtype=np.float32)
    if 'node_labels' in names:
        node_labels = read_file(folder, prefix, 'node_labels', np.int32)
        if len(node_labels.shape) >= 1:
            # node_labels = np.expand_dims(node_labels, -1)
            node_labels = node_labels.squeeze()
        node_labels = node_labels - node_labels.min()
        node_labels = np.eye(node_labels.max()+1)[node_labels]
        # node_labels = torch.cat(node_labels, dim=-1).to(torch.float)
    x = cat([node_attributes, node_labels])
    
    edge_attributes, edge_labels = None, None
    if 'edge_attributes' in names:
        edge_attributes = read_file(folder, prefix, 'edge_attributes')
    if 'edge_labels' in names:
        edge_labels = read_file(folder, prefix, 'edge_labels', np.int32)
        if len(edge_labels.shape) >= 1:
            edge_labels = edge_labels.squeeze()
        edge_labels = edge_labels - edge_labels.min()
        edge_labels = np.eye(edge_labels.max()+1)[edge_labels]
    edge_attr = cat([edge_attributes, edge_labels])
    
    y = None
    if 'graph_attributes' in names:  # Regression problem.
        y = read_file(folder, prefix, 'graph_attributes')
    elif 'graph_labels' in names:  # Classification problem.
        y = read_file(folder, prefix, 'graph_labels', np.int32)
        _, y = np.unique(y, return_inverse=True)
    
    num_nodes = edge_index.max().item() + 1 if x is None else x.shape[0]
    edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
    # edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes,
    #                                  num_nodes)
    
    #data = Graph(edge_index=edge_index, edge_feat=edge_attr, node_feat=x, node_label=y)
    graph, slices = split(edge_index, batch, x, edge_attr, y)
    
    return graph, slices


def remove_self_loops(edge_index, edge_attr=None):
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
    

def read_file(folder, prefix, name, dtype=None):
    path = osp.join(folder, f'{prefix}_{name}.txt')
    return read_txt_array(path, sep=',', dtype=dtype)


def cat(seq):
    seq = [item for item in seq if item is not None]
    seq = [np.expand_dims(item, axis=-1) if len(item.shape) == 1 else item for item in seq]
    return np.concatenate(seq, axis=-1) if len(seq) > 0 else None


def split(edge_index, batch, x=None, edge_attr=None, y=None):
    node_slice = np.bincount(batch).cumsum()
    node_slice = np.concatenate([np.array([0]), node_slice])
    
    row, _ = edge_index
    edge_slice = np.bincount(batch[row]).cumsum()
    edge_slice = np.concatenate([np.array([0]), edge_slice])
    
    # Edge indices should start at zero for every graph.
    edge_index -= np.expand_dims(node_slice[batch[row]], axis=0)
    
    slices = {'edge_index': edge_slice}
    if x is not None:
        slices['x'] = node_slice
    # else:
    #     # Imitate `collate` functionality:
    #     num_nodes = torch.bincount(batch).tolist()
    #     num_nodes = batch.numel()
    if edge_attr is not None:
        slices['edge_attr'] = edge_slice
    if y is not None:
        if y.shape[0] == batch.shape[0]:
            slices['y'] = node_slice
        else:
            slices['y'] = np.arange(0, batch[-1] + 2, dtype=np.int32)
    graph = Graph(x=x, edge_index=edge_index, edge_feat=edge_attr, y=y).tensor()
    return graph, slices

