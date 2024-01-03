from typing import Optional, Callable

import os.path as osp

import tensorlayerx as tlx
import numpy as np

from gammagl.utils import coalesce
from gammagl.data import InMemoryDataset, download_url, Graph
from gammagl.utils.loop import remove_self_loops


class WikipediaNetwork(InMemoryDataset):
    r"""The Wikipedia networks introduced in the
    `"Multi-scale Attributed Node Embedding"
    <https://arxiv.org/abs/1909.13021>`_ paper.
    Nodes represent web pages and edges represent hyperlinks between them.
    Node features represent several informative nouns in the Wikipedia pages.
    The task is to predict the average daily traffic of the web page.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"chameleon"`,
            :obj:`"crocodile"`, :obj:`"squirrel"`).
        geom_gcn_preprocess (bool): If set to :obj:`True`, will load the
            pre-processed data as introduced in the `"Geom-GCN: Geometric
            Graph Convolutional Networks" <https://arxiv.org/abs/2002.05287>_`,
            in which the average monthly traffic of the web page is converted
            into five categories to predict.
            If set to :obj:`True`, the dataset :obj:`"crocodile"` is not
            available.
        transform (callable, optional): A function/transform that takes in an
            :obj:`gammagl.data.Graph` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`gammagl.data.Graph` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)

    """

    raw_url = 'https://graphmining.ai/datasets/ptg/wiki'
    processed_url = ('https://raw.githubusercontent.com/graphdml-uiuc-jlu/'
                     'geom-gcn/f1fc0d14b3b019c562737240d06ec83b07d16a8f')

    def __init__(self, root: str = None, name: str = 'chameleon', geom_gcn_preprocess: bool = True,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.name = name.lower()
        self.geom_gcn_preprocess = geom_gcn_preprocess
        assert self.name in ['chameleon', 'crocodile', 'squirrel']
        if geom_gcn_preprocess and self.name == 'crocodile':
            raise AttributeError("The dataset 'crocodile' is not available in "
                                 "case 'geom_gcn_preprocess=True'")
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = self.load_data(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        if self.geom_gcn_preprocess:
            return osp.join(self.root, self.name, 'geom_gcn', 'raw')
        else:
            return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        if self.geom_gcn_preprocess:
            return osp.join(self.root, self.name, 'geom_gcn', 'processed')
        else:
            return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> str:
        if self.geom_gcn_preprocess:
            return (['out1_node_feature_label.txt', 'out1_graph_edges.txt'] +
                    [f'{self.name}_split_0.6_0.2_{i}.npz' for i in range(10)])
        else:
            return f'{self.name}.npz'

    @property
    def processed_file_names(self) -> str:
        return tlx.BACKEND + '_data.pt'

    def download(self):
        if self.geom_gcn_preprocess:
            for filename in self.raw_file_names[:2]:
                url = f'{self.processed_url}/new_data/{self.name}/{filename}'
                download_url(url, self.raw_dir)
            for filename in self.raw_file_names[2:]:
                url = f'{self.processed_url}/splits/{filename}'
                download_url(url, self.raw_dir)
        else:
            download_url(f'{self.raw_url}/{self.name}.npz', self.raw_dir)

    def process(self):
        if self.geom_gcn_preprocess:
            with open(self.raw_paths[0], 'r') as f:
                data = f.read().split('\n')[1:-1]
                x = [[float(v) for v in r.split('\t')[1].split(',')] for r in data]
                x = np.array(x, dtype=np.float32)
                y = [int(r.split('\t')[2]) for r in data]
                y = np.array(y, dtype=np.int64)

            with open(self.raw_paths[1], 'r') as f:
                data = f.read().split('\n')[1:-1]
                data = [[int(v) for v in r.split('\t')] for r in data]
                edge_index = np.ascontiguousarray(np.array(data, dtype=np.int64).T)
                edge_index = coalesce(edge_index)
            # with open(self.raw_paths[0], 'r') as f:
            #     data = f.read().split('\n')[1:-1]
            # x = [[float(v) for v in r.split('\t')[1].split(',')] for r in data]
            # x = np.array(x, dtype=np.float32)
            # y = [int(r.split('\t')[2]) for r in data]
            # y = np.array(y, dtype=np.int64)
            #
            # with open(self.raw_paths[1], 'r') as f:
            #     data = f.read().split('\n')[1:-1]
            #     data = [[int(v) for v in r.split('\t')] for r in data]
            # edge_index = np.ascontiguousarray(np.array(data, dtype=np.int64).T)
            # edge_index, _ = coalesce(edge_index, None, x.size, x.size)

            train_masks, val_masks, test_masks = [], [], []
            for f in self.raw_paths[2:]:
                tmp = np.load(f)
                train_masks += [tmp['train_mask'].astype(np.bool_)]
                val_masks += [tmp['val_mask'].astype(np.bool_)]
                test_masks += [tmp['test_mask'].astype(np.bool_)]
            train_mask = np.concatenate(train_masks)
            val_mask = np.concatenate(val_masks)
            test_mask = np.concatenate(test_masks)
            data = Graph(x=x, edge_index=edge_index, y=y, train_mask=train_mask,
                         val_mask=val_mask, test_mask=test_mask)

        else:
            data = np.load(self.raw_paths[0], 'r', allow_pickle=True)
            x = data['features'].astype(np.float32)
            edge_index = data['edges'].astype(np.int64)
            edge_index = np.ascontiguousarray(edge_index.T)
            edge_index, _ = remove_self_loops(edge_index)
            edge_index = coalesce(edge_index)
            y = data['target'].astype(np.float32)

            data = Graph(x=x, edge_index=edge_index, y=y)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save_data(self.collate([data]), self.processed_paths[0])
