import os.path as osp
from typing import Callable, Optional

import tensorlayerx as tlx
from gammagl.data import InMemoryDataset, download_url, Graph
import numpy as np

class DeezerEurope(InMemoryDataset):
    url = 'https://graphmining.ai/datasets/ptg/deezer_europe.npz'

    def __init__(self, root: str, transform: Optional[Callable] = None, pre_transform: Optional[Callable] = None, pre_filter: Optional[Callable] = None) -> None:
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = self.load_data(self.processed_paths[0])

    @property
    def raw_file_names(self) -> str:
        return 'deezer_europe.npz'

    @property
    def processed_file_names(self) -> str:
        return tlx.BACKEND + '_data.npy'

    def download(self) -> None:
        download_url(self.url, self.raw_dir)

    def process(self) -> None:
        data = np.load(self.raw_paths[0], allow_pickle=True)
        x = data['features'].astype(np.float32)
        y = data['target'].astype(np.float32)
        edge_index = data['edges'].astype(np.int64)
        edge_index = edge_index.T  

        data = Graph(x=x, y=y, edge_index=edge_index)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save_data(self.collate([data]), self.processed_paths[0])
