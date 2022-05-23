import os
import os.path as osp
import numpy as np
import scipy.sparse as sp
from gammagl.data import extract_zip, download_url, InMemoryDataset, Graph
from gammagl.utils import coalesce


class Reddit(InMemoryDataset):
    r"""The Reddit dataset from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper, containing
    Reddit posts belonging to different communities.

    Parameters
    ----------
    root: string
        Root directory where the dataset should be saved.
    transform: callable, optional
        A function/transform that takes in an
        :obj:`gammagl.data.Graph` object and returns a transformed
        version. The data object will be transformed before every access.
        (default: :obj:`None`)
    pre_transform: callable, optional
        A function/transform that takes in
        an :obj:`gammagl.data.Graph` object and returns a
        transformed version. The data object will be transformed before
        being saved to disk. (default: :obj:`None`)
    """

    url = 'https://data.dgl.ai/dataset/reddit.zip'

    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = self.load_data(self.processed_paths[0])

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
        x = np.array(data['feature'], dtype=np.float32)
        y = np.array(data['label'], np.int32)
        split = np.array(data['node_types'])

        adj = sp.load_npz(osp.join(self.raw_dir, 'reddit_graph.npz'))

        edge_index = np.array([adj.row, adj.col], dtype=np.int64)

        edge_index = coalesce(edge_index, None, x.shape[0])

        data = Graph(edge_index=edge_index, x=x, y=y)

        data.train_mask = split == 1
        data.val_mask = split == 2
        data.test_mask = split == 3

        data = data if self.pre_transform is None else self.pre_transform(data)
        self.save_data(self.collate([data]), self.processed_paths[0])
