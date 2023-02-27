import os
import os.path as osp
from itertools import product
from typing import Callable, List, Optional

import numpy as np
import scipy.sparse as sp
import tensorlayerx as tlx

from gammagl.data import (HeteroGraph, InMemoryDataset, download_url,
                                  extract_zip)


class IMDB(InMemoryDataset):
    r"""A subset of the Internet Movie Database (IMDB), as collected in the
    `"MAGNN: Metapath Aggregated Graph Neural Network for Heterogeneous Graph
    Embedding" <https://arxiv.org/abs/2002.01680>`_ paper.
    IMDB is a heterogeneous graph containing three types of entities - movies
    (4,278 nodes), actors (5,257 nodes), and directors (2,081 nodes).
    The movies are divided into three classes (action, comedy, drama) according
    to their genre.
    Movie features correspond to elements of a bag-of-words representation of
    its plot keywords.
    
    Parameters
    ----------
    root: string
        Root directory where the dataset should be saved.
    transform: callable, optional
        A function/transform that takes in an
        :obj:`gammagl.data.HeteroGraph` object and returns a
        transformed version. The data object will be transformed before
        every access. (default: :obj:`None`)
    pre_transform: callable, optional
        A function/transform that takes in
        an :obj:`gammagl.data.HeteroGraph` object and returns a
        transformed version. The data object will be transformed before
        being saved to disk. (default: :obj:`None`)
    """

    url = 'https://www.dropbox.com/s/g0btk9ctr1es39x/IMDB_processed.zip?dl=1'

    def __init__(self, root: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = self.load_data(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return [
            'adjM.npz', 'features_0.npz', 'features_1.npz', 'features_2.npz',
            'labels.npy', 'train_val_test_idx.npz'
        ]

    @property
    def processed_file_names(self) -> str:
        return tlx.BACKEND+'data.pt'

    def download(self):
        path = download_url(self.url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.remove(path)

    def process(self):
        data = HeteroGraph()

        node_types = ['movie', 'director', 'actor']
        for i, node_type in enumerate(node_types):
            x = sp.load_npz(osp.join(self.raw_dir, f'features_{i}.npz'))
            data[node_type].x = tlx.convert_to_tensor(x.todense(), dtype=tlx.float32)

        y = np.load(osp.join(self.raw_dir, 'labels.npy'))
        data['movie'].y = tlx.convert_to_tensor(y, dtype=tlx.int64)

        split = np.load(osp.join(self.raw_dir, 'train_val_test_idx.npz'))
        for name in ['train', 'val', 'test']:
            idx = split[f'{name}_idx']
            mask = np.zeros(data['movie'].num_nodes, dtype=np.bool_)
            mask[idx] = True
            data['movie'][f'{name}_mask'] = tlx.convert_to_tensor(mask, dtype=tlx.bool)

        s = {}
        N_m = data['movie'].num_nodes
        N_d = data['director'].num_nodes
        N_a = data['actor'].num_nodes
        s['movie'] = (0, N_m)
        s['director'] = (N_m, N_m + N_d)
        s['actor'] = (N_m + N_d, N_m + N_d + N_a)

        A = sp.load_npz(osp.join(self.raw_dir, 'adjM.npz'))
        for src, dst in product(node_types, node_types):
            A_sub = A[s[src][0]:s[src][1], s[dst][0]:s[dst][1]].tocoo()
            if A_sub.nnz > 0:
                row = tlx.convert_to_tensor(A_sub.row, dtype=tlx.int64)
                col = tlx.convert_to_tensor(A_sub.col, dtype=tlx.int64)
                data[src, dst].edge_index = tlx.stack([row, col], axis=0)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save_data(self.collate([data]), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'