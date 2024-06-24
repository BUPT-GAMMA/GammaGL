from typing import Callable, Optional
import os
import numpy as np
import tensorlayerx as tlx

from gammagl.data import Graph, InMemoryDataset, download_url

class FacebookPagePage(InMemoryDataset):
    r"""The Facebook Page-Page network dataset introduced in the
    `"Multi-scale Attributed Node Embedding"
    <https://arxiv.org/abs/1909.13021>`_ paper.
    Nodes represent verified pages on Facebook and edges are mutual likes.
    It contains 22,470 nodes, 342,004 edges, 128 node features and 4 classes.

    Args:
        root (str): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`gammagl.data.Graph` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`gammagl.data.Graph` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)
    """

    url = 'https://graphmining.ai/datasets/ptg/facebook.npz'

    def __init__(
        self,
        root: str = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        super().__init__(root, transform, pre_transform, force_reload=force_reload)
        self.data, self.slices=self.load_data(self.processed_paths[0])

    @property
    def raw_file_names(self) -> str:
        return 'facebook.npz'

    @property
    def processed_file_names(self) -> str:
        return tlx.BACKEND + '_data.pt'

    def download(self) -> None:
        download_url(self.url, self.raw_dir)

    def process(self) -> None:
        data = np.load(self.raw_paths[0], 'r', allow_pickle=True)
        x = tlx.convert_to_tensor(data['features'], dtype=tlx.float32)
        y = tlx.convert_to_tensor(data['target'], dtype=tlx.int64)
        edge_index = tlx.convert_to_tensor(data['edges'], dtype=tlx.int64)
        edge_index = edge_index.T

        data = Graph(x=x, edge_index=edge_index, y=y)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save_data(self.collate([data]), self.processed_paths[0])



