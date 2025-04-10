import numpy as np
import tensorlayerx as tlx
import os.path as osp
from typing import Callable, List, Optional
from gammagl.data import download_url
from gammagl.data import InMemoryDataset
from gammagl.data import Graph
try:
    import cPickle as pickle
except ImportError:
    import pickle


class ADDataset(InMemoryDataset):
    r"""Some annomaly detection datasets (E.g inj_cora, inj_amazon).
    Compared to the source data set, they are all artificially injected with anomaly.
    This class provides a dataset with a version of the .npz suffix.
    Original version: `<https://github.com/pygod-team/data>`
    npz version: `<https://github.com/SharkRemW/data/tree/main>`

    Parameters
    ----------
    root: str
        Root directory where the dataset should be saved.
    name: str
        Dataset name. (:obj:`"inj_cora"`, :obj:`"inj_amazon"`, :obj:`"reddit"`)
    transform: callable, optional
        A function/transform that takes in a
        :obj:`gammagl.data.Graph` object and returns a transformed
        version. The data object will be transformed before every access.
        (default: :obj:`None`)
    pre_transform: callable, optional
        A function/transform that takes in
        an :obj:`gammagl.data.Graph` object and returns a
        transformed version. The data object will be transformed before
        being saved to disk. (default: :obj:`None`)
    force_reload (bool, optional): Whether to re-process the dataset.
        (default: :obj:`False`)

    """


    url = 'https://github.com/SharkRemW/data/raw/main/processed'
    def __init__(self, root: str = None, name: str = 'inj_cora', 
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 force_reload: bool = False):
        self.name = name.lower()
        super().__init__(root, transform, pre_transform, force_reload = force_reload)
        self.data, self.slices = self.load_data(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        return [f'{self.name.lower()}.npz']

    @property
    def processed_file_names(self) -> str:
        return tlx.BACKEND + '_data.pt'

    def download(self):
        for name in self.raw_file_names:
            download_url(f'{self.url}/{name}', self.raw_dir)

    def process(self):
        data = np.load(osp.join(self.raw_dir, self.name + '.npz'))
        
        edge_index = data['edge_index']
        x = data['x']
        y = data['y']

        data = Graph(edge_index=edge_index, x=x, y=y).tensor()
        data = data if self.pre_transform is None else self.pre_transform(data)
        self.save_data(self.collate([data]), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name}()'
    

# data = ADDataset('./', 'books')
# print(data)
# print(data[0])

