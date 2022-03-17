import os
import os.path as osp
import pickle

import tensorlayerx as tlx
import numpy as np
import scipy.sparse as sp

from gammagl.data import extract_zip, download_url, InMemoryDataset, Graph
from gammagl.sparse import CSRAdj


class Reddit(InMemoryDataset):
    r"""The Reddit dataset from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper, containing
    Reddit posts belonging to different communities.

    Args:
        root (string): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    url = 'https://data.dgl.ai/dataset/reddit.zip'

    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        # self.data, self.slices = torch.load(self.processed_paths[0])
        with open(self.processed_paths[0], 'rb') as f:
            self.data, self.slices = pickle.load(f)

    @property
    def raw_file_names(self):
        return ['reddit_data.npz', 'reddit_graph.npz']

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        path = download_url(self.url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.unlink(path)

    def process(self):
        data = np.load(osp.join(self.raw_dir, 'reddit_data.npz'))
        x = tlx.convert_to_tensor(data['feature'], dtype=tlx.float32)
        y = (data['label'], tlx.int32)
        split = tlx.convert_to_tensor(data['node_types'])

        adj = sp.load_npz(osp.join(self.raw_dir, 'reddit_graph.npz'))
        row = tlx.convert_to_tensor(adj.row, dtype=tlx.int32)
        col = tlx.convert_to_tensor(adj.col, dtype=tlx.int32)
        edge_index = tlx.stack([row, col], axis=0)
        # edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))
        data = Graph(edge_index=edge_index, x=x, y=y)
        # data = Data(x=x, edge_index=edge_index, y=y)
        data.train_mask = split == 1
        data.val_mask = split == 2
        data.test_mask = split == 3

        data = data if self.pre_transform is None else self.pre_transform(data)

        # torch.save(self.collate([data]), self.processed_paths[0])

    def process(self):
        data = np.load(osp.join(self.raw_dir, 'reddit_data.npz'))
        x = data['feature']
        y = data['label']
        split = tlx.convert_to_tensor(data['node_types'])

        adj = sp.load_npz(osp.join(self.raw_dir, 'reddit_graph.npz'))
        row = tlx.convert_to_tensor(adj.row, dtype=tlx.int32)
        col = tlx.convert_to_tensor(adj.col, dtype=tlx.int32)
        # 如果图有处理的edge的操作，这下面就可以删去
        edge = np.array([col, row])
        ind = np.argsort(edge[1], axis=0)
        edge = np.array(edge.T[ind])
        # cur = CSRAdj.from_edges(col, row, 232965)
        # cur.
        data = Graph(edge_index=edge.T, x=x, y=y)
        # data = Data(x=x, edge_index=edge_index, y=y)
        data.train_mask = split == 1
        data.val_mask = split == 2
        data.test_mask = split == 3

        data = data if self.pre_transform is None else self.pre_transform(data)
        with open(self.processed_paths[0], 'wb') as f:
            pickle.dump(self.collate([data]), f)

        # torch.save(self.collate([data]), self.processed_paths[0])

