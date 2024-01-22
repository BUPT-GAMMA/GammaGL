import numpy as np
import tensorlayerx as tlx
import os.path as osp
import json
import warnings
from itertools import chain
from typing import Callable, List, Optional
from gammagl.data import download_url, InMemoryDataset, Graph
from gammagl.utils import to_undirected

class WikiCS(InMemoryDataset):
    r"""The semi-supervised Wikipedia-based dataset from the
    `"Wiki-CS: A Wikipedia-Based Benchmark for Graph Neural Networks"
    <https://arxiv.org/abs/2007.02901>`_ paper, containing 11,701 nodes,
    216,123 edges, 10 classes and 20 different training splits.

    Parameters
    ----------
    root: str, optional
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
    is_undirected: bool, optional
        Whether the graph is undirected.
        (default: :obj:`True`)
    force_reload (bool, optional): Whether to re-process the dataset.
        (default: :obj:`False`)
        
    """

    url = 'https://github.com/pmernyei/wiki-cs-dataset/raw/master/dataset'

    def __init__(self, root: str = None, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 is_undirected: Optional[bool] = None,
                 force_reload: bool = False):
        if is_undirected is None:
            warnings.warn(
                f"The {self.__class__.__name__} dataset now returns an "
                f"undirected graph by default. Please explicitly specify "
                f"'is_undirected=False' to restore the old behavior.")
            is_undirected = True
        self.is_undirected = is_undirected
        super().__init__(root, transform, pre_transform, force_reload = force_reload)
        self.data, self.slices = self.load_data(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, 'wikics', 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'wikics', 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        return ['data.json']

    @property
    def processed_file_names(self) -> str:
        return tlx.BACKEND + '_data_undirected.pt' if self.is_undirected else '_data.pt'

    def download(self):
        for name in self.raw_file_names:
            download_url(f'{self.url}/{name}', self.raw_dir)

    def process(self):
        with open(self.raw_paths[0], 'r') as f:
            data = json.load(f)

        x = tlx.convert_to_tensor(data['features'], dtype=tlx.float32)
        y = tlx.convert_to_tensor(data['labels'], dtype=tlx.int64)

        edges = [[(i, j) for j in js] for i, js in enumerate(data['links'])]
        edges = list(chain(*edges))
        edge_index = np.ascontiguousarray(np.array(edges, dtype=np.int64).T)
        if self.is_undirected:
            edge_index = to_undirected(tlx.convert_to_tensor(edge_index), num_nodes=tlx.get_tensor_shape(x[0]))

        train_mask = np.ascontiguousarray(np.array(data['train_masks'], dtype=np.bool).T)
        val_mask = np.ascontiguousarray(np.array(data['val_masks'], dtype=np.bool).T)
        test_mask = np.ascontiguousarray(np.array(data['test_mask'], dtype=np.bool).T)
        stopping_mask = np.ascontiguousarray(np.array(data['stopping_masks'], dtype=np.bool).T)

        data = Graph(x=x, y=y, edge_index=edge_index, train_mask=train_mask,
                    val_mask=val_mask, test_mask=test_mask,
                    stopping_mask=stopping_mask)

        if self.pre_transform is not None:
            data = self.pre_transform(data)
        self.save_data(self.collate([data]), self.processed_paths[0])