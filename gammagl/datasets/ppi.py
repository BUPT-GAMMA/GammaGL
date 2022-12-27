import os
import json
import os.path as osp
from gammagl.data import (InMemoryDataset, download_url,
                          extract_zip)
import tensorlayerx as tlx
import numpy as np
from itertools import product
from gammagl.data import Graph
from gammagl.utils import remove_self_loops


class PPI(InMemoryDataset):
    url = 'https://data.dgl.ai/dataset/ppi.zip'

    def __init__(self, root, split='train', transform=None, pre_transform=None,
                 pre_filter=None):

        assert split in ['train', 'val', 'valid', 'test']

        super().__init__(root, transform, pre_transform, pre_filter)

        if split == 'train':
            self.data, self.slices = self.load_data(self.processed_paths[0])
        elif split == 'val' or split == 'valid':
            self.data, self.slices = self.load_data(self.processed_paths[1])
        elif split == 'test':
            self.data, self.slices = self.load_data(self.processed_paths[2])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, 'ppi', 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'ppi', 'processed')

    @property
    def raw_file_names(self):
        splits = ['train', 'valid', 'test']
        files = ['feats.npy', 'graph_id.npy', 'graph.json', 'labels.npy']
        return [f'{split}_{name}' for split, name in product(splits, files)]

    @property
    def processed_file_names(self):
        return [f'{tlx.BACKEND}_{postfix}' for postfix in ['train.pt', 'val.pt', 'test.pt']]

    def download(self):
        path = download_url(self.url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.remove(path)

    def process(self):
        for s, split in enumerate(['train', 'valid', 'test']):
            path = osp.join(self.raw_dir, f'{split}_graph.json')
            with open(path, 'r') as f:
                graph_json = json.load(f)

            links = graph_json['links']
            edges = [(dict['source'], dict['target']) for dict in links]

            x = np.load(osp.join(self.raw_dir, f'{split}_feats.npy'))
            tlx.convert_to_tensor(x, dtype=tlx.float32)

            y = np.load(osp.join(self.raw_dir, f'{split}_labels.npy'))
            tlx.convert_to_tensor(y, dtype=tlx.float32)

            graph_list = []
            path = osp.join(self.raw_dir, f'{split}_graph_id.npy')
            idx = np.load(path).astype(np.int64)
            idx = idx - idx.min()

            for i in range(idx.max() + 1):
                mask = idx == i
                mask_index = np.nonzero(mask)[0]
                edge_index = []
                for edge in edges:
                    if mask_index[0] <= edge[0] <= mask_index[-1]:
                        edge_index.append(edge)

                edge_index = np.array(edge_index, dtype=int).T
                edge_index = edge_index - np.min(edge_index)
                edge_index, _ = remove_self_loops(edge_index)
                graph = Graph(edge_index=edge_index, x=x[mask], y=y[mask])
                graph_list.append(graph)
            self.save_data(self.collate(graph_list), self.processed_paths[s])