import os
import os.path as osp
import pickle
import numpy as np
import networkx as nx
import tensorlayerx as tlx
from typing import Callable, List, Optional

from gammagl.data import (
    Graph,
    InMemoryDataset,
    download_url,
)


def _adj_to_edge_index_and_attr(adj, num_edge_types=2):
    r"""Convert a dense adjacency matrix to edge_index and edge_attr.

    Parameters
    ----------
    adj : numpy.ndarray
        Dense adjacency matrix of shape ``(n, n)``.
    num_edge_types : int
        Number of edge types. Default is 2 (no-edge=0, edge=1).

    Returns
    -------
    edge_index : numpy.ndarray
        Edge indices of shape ``(2, E)``.
    edge_attr : numpy.ndarray
        Edge attributes of shape ``(E, num_edge_types)``.
    """
    np.fill_diagonal(adj, 0)
    src, dst = np.nonzero(adj)
    edge_index = np.stack([src, dst], axis=0).astype(np.int64)
    # One-hot: index 0 = no-edge (not stored), index 1 = edge present
    edge_attr = np.zeros((len(src), num_edge_types), dtype=np.float32)
    edge_attr[:, 1] = 1.0
    return edge_index, edge_attr


class SpectreGraphDataset(InMemoryDataset):
    r"""Base class for synthetic graph datasets from the SPECTRE benchmark.

    Downloads pre-generated graphs stored as NetworkX pickle files and
    converts them to GammaGL :class:`Graph` objects.  Three splits
    (train / val / test) are provided directly in the downloaded file.

    Subclassed by :class:`PlanarGraphDataset`, :class:`TreeGraphDataset`,
    and :class:`SBMGraphDataset`.

    Parameters
    ----------
    root : str, optional
        Root directory where the dataset should be saved.
    name : str
        Dataset name, one of ``'planar'``, ``'tree'``, ``'sbm'``.
    split : str
        Which split to load: ``'train'``, ``'val'``, or ``'test'``.
    transform : callable, optional
        A transform applied to each :class:`Graph` on access.
    pre_transform : callable, optional
        A transform applied during processing before saving.
    pre_filter : callable, optional
        A filter applied during processing.
    force_reload : bool
        Whether to re-process the dataset.
    """

    urls = {
        'planar': 'https://raw.githubusercontent.com/AndreasBergmeister/'
                  'graph-generation/main/data/planar.pkl',
        'tree': 'https://raw.githubusercontent.com/AndreasBergmeister/'
                'graph-generation/main/data/tree.pkl',
        'sbm': 'https://raw.githubusercontent.com/AndreasBergmeister/'
               'graph-generation/main/data/sbm.pkl',
    }

    def __init__(self, root: Optional[str] = None, name: str = 'planar',
                 split: str = 'train',
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 force_reload: bool = False):
        assert name in self.urls, f"Unknown dataset name: {name}"
        assert split in ('train', 'val', 'test'), f"Unknown split: {split}"
        self.name = name
        self.split = split

        super().__init__(root, transform, pre_transform, pre_filter,
                         force_reload=force_reload)

        split_idx = {'train': 0, 'val': 1, 'test': 2}[split]
        self.data, self.slices = self.load_data(self.processed_paths[split_idx])

    @property
    def raw_file_names(self) -> List[str]:
        return [f'{self.name}.pkl']

    @property
    def processed_file_names(self) -> List[str]:
        return [
            tlx.BACKEND + '_train.pt',
            tlx.BACKEND + '_val.pt',
            tlx.BACKEND + '_test.pt',
        ]

    def download(self):
        download_url(self.urls[self.name], self.raw_dir)

    def process(self):
        pkl_path = osp.join(self.raw_dir, f'{self.name}.pkl')
        with open(pkl_path, 'rb') as f:
            raw_data = pickle.load(f)

        for split_name, split_idx in [('train', 0), ('val', 1), ('test', 2)]:
            graphs_nx = raw_data[split_name]
            data_list = []

            for g in graphs_nx:
                adj = nx.to_numpy_array(g).astype(np.float32)
                n = adj.shape[0]
                edge_index, edge_attr = _adj_to_edge_index_and_attr(adj)

                x = np.ones((n, 1), dtype=np.float32)
                y = np.zeros((1, 0), dtype=np.float32)

                data = Graph(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    y=y,
                    n_nodes=np.array([n], dtype=np.int64),
                    to_tensor=True,
                )

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)

            collated_data, slices = self.collate(data_list)
            self.save_data(
                (collated_data, slices),
                self.processed_paths[split_idx],
            )

    def __repr__(self) -> str:
        return f'{self.name.capitalize()}GraphDataset({len(self)})'


class PlanarGraphDataset(SpectreGraphDataset):
    r"""The Planar graph dataset from the SPECTRE benchmark.

    Contains connected planar graphs with 64 nodes each. Used for
    evaluating graph generation models.

    From `"Spectre: Spectral conditioning helps to overcome the
    expressivity limits of one-shot graph generators"
    <https://proceedings.mlr.press/v162/martinkus22a.html>`_ (ICML 2022).

    Parameters
    ----------
    root : str, optional
        Root directory where the dataset should be saved.
    split : str
        Which split to load: ``'train'``, ``'val'``, or ``'test'``.
    transform : callable, optional
        A transform applied to each :class:`Graph` on access.
    pre_transform : callable, optional
        A transform applied during processing before saving.
    pre_filter : callable, optional
        A filter applied during processing.
    force_reload : bool
        Whether to re-process the dataset.
    """

    def __init__(self, root: Optional[str] = None, split: str = 'train',
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 force_reload: bool = False):
        super().__init__(root, name='planar', split=split,
                         transform=transform, pre_transform=pre_transform,
                         pre_filter=pre_filter, force_reload=force_reload)

    def download(self):
        super().download()

    def process(self):
        super().process()


class TreeGraphDataset(SpectreGraphDataset):
    r"""The Tree graph dataset from the HSpectre benchmark.

    Contains tree graphs (connected acyclic graphs) with 64 nodes each.

    From `"Efficient and scalable graph generation through iterative
    local expansion" <https://arxiv.org/abs/2312.11529>`_ (ICLR 2023).

    Parameters
    ----------
    root : str, optional
        Root directory where the dataset should be saved.
    split : str
        Which split to load: ``'train'``, ``'val'``, or ``'test'``.
    transform : callable, optional
        A transform applied to each :class:`Graph` on access.
    pre_transform : callable, optional
        A transform applied during processing before saving.
    pre_filter : callable, optional
        A filter applied during processing.
    force_reload : bool
        Whether to re-process the dataset.
    """

    def __init__(self, root: Optional[str] = None, split: str = 'train',
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 force_reload: bool = False):
        super().__init__(root, name='tree', split=split,
                         transform=transform, pre_transform=pre_transform,
                         pre_filter=pre_filter, force_reload=force_reload)

    def download(self):
        super().download()

    def process(self):
        super().process()


class SBMGraphDataset(SpectreGraphDataset):
    r"""The Stochastic Block Model (SBM) graph dataset from the SPECTRE benchmark.

    Contains synthetic clustering graphs where nodes within the same
    cluster have a higher probability of being connected.  Node count
    ranges from 44 to 187.

    From `"Spectre: Spectral conditioning helps to overcome the
    expressivity limits of one-shot graph generators"
    <https://proceedings.mlr.press/v162/martinkus22a.html>`_ (ICML 2022).

    Parameters
    ----------
    root : str, optional
        Root directory where the dataset should be saved.
    split : str
        Which split to load: ``'train'``, ``'val'``, or ``'test'``.
    transform : callable, optional
        A transform applied to each :class:`Graph` on access.
    pre_transform : callable, optional
        A transform applied during processing before saving.
    pre_filter : callable, optional
        A filter applied during processing.
    force_reload : bool
        Whether to re-process the dataset.
    """

    def __init__(self, root: Optional[str] = None, split: str = 'train',
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 force_reload: bool = False):
        super().__init__(root, name='sbm', split=split,
                         transform=transform, pre_transform=pre_transform,
                         pre_filter=pre_filter, force_reload=force_reload)

    def download(self):
        super().download()

    def process(self):
        super().process()


class Comm20GraphDataset(InMemoryDataset):
    r"""The Community-20 graph dataset from the SPECTRE benchmark.

    Contains 200 small community-structured graphs.  The raw data is a
    PyTorch ``.pt`` file storing dense adjacency matrices, which are
    split into train / val / test with a fixed random seed.

    Parameters
    ----------
    root : str, optional
        Root directory where the dataset should be saved.
    split : str
        Which split to load: ``'train'``, ``'val'``, or ``'test'``.
    transform : callable, optional
        A transform applied to each :class:`Graph` on access.
    pre_transform : callable, optional
        A transform applied during processing before saving.
    pre_filter : callable, optional
        A filter applied during processing.
    force_reload : bool
        Whether to re-process the dataset.
    """

    URL = ('https://raw.githubusercontent.com/KarolisMart/SPECTRE/'
           'main/data/community_12_21_100.pt')
    NUM_GRAPHS = 100

    def __init__(self, root: Optional[str] = None, split: str = 'train',
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 force_reload: bool = False):
        assert split in ('train', 'val', 'test'), f"Unknown split: {split}"
        self.name = 'comm20'
        self.split = split
        super().__init__(root, transform, pre_transform, pre_filter,
                         force_reload=force_reload)
        split_idx = {'train': 0, 'val': 1, 'test': 2}[split]
        self.data, self.slices = self.load_data(self.processed_paths[split_idx])

    @property
    def raw_file_names(self) -> List[str]:
        return ['community_12_21_100.pt']

    @property
    def processed_file_names(self) -> List[str]:
        return [
            tlx.BACKEND + '_train.pt',
            tlx.BACKEND + '_val.pt',
            tlx.BACKEND + '_test.pt',
        ]

    def download(self):
        download_url(self.URL, self.raw_dir)

    def process(self):
        import torch
        pt_path = osp.join(self.raw_dir, 'community_12_21_100.pt')
        payload = torch.load(pt_path, map_location='cpu', weights_only=False)
        adjs = payload[0]  # list of dense adjacency tensors

        rng = np.random.default_rng(1234)
        indices = rng.permutation(self.NUM_GRAPHS)
        test_len = int(round(self.NUM_GRAPHS * 0.2))
        train_len = int(round((self.NUM_GRAPHS - test_len) * 0.8))
        val_len = self.NUM_GRAPHS - train_len - test_len

        split_indices = {
            'train': indices[:train_len],
            'val': indices[train_len:train_len + val_len],
            'test': indices[train_len + val_len:],
        }

        for split_name, split_idx in [('train', 0), ('val', 1), ('test', 2)]:
            data_list = []
            for i in split_indices[split_name]:
                adj = adjs[i].numpy().astype(np.float32)
                n = adj.shape[0]
                edge_index, edge_attr = _adj_to_edge_index_and_attr(adj)

                x = np.ones((n, 1), dtype=np.float32)
                y = np.zeros((1, 0), dtype=np.float32)

                data = Graph(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    y=y,
                    n_nodes=np.array([n], dtype=np.int64),
                    to_tensor=True,
                )
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                data_list.append(data)

            collated_data, slices = self.collate(data_list)
            self.save_data(
                (collated_data, slices),
                self.processed_paths[split_idx],
            )

    def __repr__(self) -> str:
        return f'Comm20GraphDataset({len(self)})'
