import os
import os.path as osp
from itertools import product
from typing import Callable, List, Optional

import numpy as np
import scipy.sparse as sp
import tensorlayerx as tlx

from gammagl.data import (HeteroGraph, InMemoryDataset, download_url,
                          extract_zip)

class AGNews(InMemoryDataset):
    r"""AGNews dataset processed for use in GNN models."""

    url = 'https://www.dropbox.com/scl/fi/m809k1xdqzf0rhdmb83jf/agnews.zip?rlkey=wrz4by7f4tvtsdte2scuiec5k&st=s3ty36oi&dl=1' 

    def __init__(self, root: str = None, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None, force_reload: bool = False):
        super().__init__(root, transform, pre_transform, force_reload=force_reload)
        self.data, self.slices = self.load_data(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return [
            'adjM.npz', 'features_0.npz', 'features_1.npz', 'features_2.npz',
            'labels.npy', 'train_val_test_idx.npz'
        ]

    @property
    def processed_file_names(self) -> str:
        return tlx.BACKEND + 'data.pt'

    def download(self):
        path = download_url(self.url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.remove(path)

    def process(self):
        data = HeteroGraph()

        node_types = ['text', 'topic', 'entity']
        for i, node_type in enumerate(node_types):
            x = sp.load_npz(osp.join(self.raw_dir, f'features_{i}.npz'))
            data[node_type].x = tlx.convert_to_tensor(x.todense(), dtype=tlx.float32)
        y = np.load(osp.join(self.raw_dir, 'labels.npy'))
        y = np.argmax(y,axis=1)
        data['text'].y = tlx.convert_to_tensor(y, dtype=tlx.int64)

        split = np.load(osp.join(self.raw_dir, 'train_val_test_idx.npz'))
        for name in ['train', 'val', 'test']:
            idx = split[f'{name}_idx']
            mask = np.zeros(data['text'].num_nodes, dtype=np.bool_)
            mask[idx] = True
            data['text'][f'{name}_mask'] = tlx.convert_to_tensor(mask, dtype=tlx.bool)


        s = {}
        N_m = data['text'].num_nodes
        N_d = data['topic'].num_nodes
        N_a = data['entity'].num_nodes
        s['text'] = (0, N_m)
        s['topic'] = (N_m, N_m + N_d)
        s['entity'] = (N_m + N_d, N_m + N_d + N_a)

        A = sp.load_npz(osp.join(self.raw_dir, 'adjM.npz')).tocsr()
        for src, dst in product(node_types, node_types):
            A_sub = A[s[src][0]:s[src][1], s[dst][0]:s[dst][1]].tocoo()
            if A_sub.nnz > 0:
                row = tlx.convert_to_tensor(A_sub.row, dtype=tlx.int64)
                col = tlx.convert_to_tensor(A_sub.col, dtype=tlx.int64)
                data[src, dst].edge_index = tlx.stack([row, col], axis=0)
                print(src+"____"+dst)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save_data(self.collate([data]), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'