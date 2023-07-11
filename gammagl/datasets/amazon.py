import os.path as osp
from typing import Callable, Optional
import tensorlayerx as tlx
from gammagl.data import InMemoryDataset, download_url
from gammagl.io.npz import read_npz


class Amazon(InMemoryDataset):
    r"""The Amazon Computers and Amazon Photo networks from the
    `"Pitfalls of Graph Neural Network Evaluation"
    <https://arxiv.org/abs/1811.05868>`_ paper.
    Nodes represent goods and edges represent that two goods are frequently
    bought together.
    Given product reviews as bag-of-words node features, the task is to
    map goods to their respective product category.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"Computers"`,
            :obj:`"Photo"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`gammagl.data.Graph` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`gammagl.data.Graph` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)

    Stats:
        .. list-table::
            :widths: 10 10 10 10 10
            :header-rows: 1

            * - Name
              - #nodes
              - #edges
              - #features
              - #classes
            * - Computers
              - 13,752
              - 491,722
              - 767
              - 10
            * - Photo
              - 7,650
              - 238,162
              - 745
              - 8
    """

    url = 'https://github.com/shchur/gnn-benchmark/raw/master/data/npz/'

    def __init__(self, root: str = None, name: str = 'computers',
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.name = name.lower()
        assert self.name in ['computers', 'photo']
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = self.load_data(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name.capitalize(), 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name.capitalize(), 'processed')

    @property
    def raw_file_names(self) -> str:
        return f'amazon_electronics_{self.name.lower()}.npz'

    @property
    def processed_file_names(self) -> str:
        return tlx.BACKEND + 'data.pt'

    def download(self):
        download_url(self.url + self.raw_file_names, self.raw_dir)

    def process(self):
        data = read_npz(self.raw_paths[0])
        data = data if self.pre_transform is None else self.pre_transform(data)
        self.save_data(self.collate([data]), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}{self.name.capitalize()}()'

# data=Amazon(root='./Amazon/',name='photo')
# data.process()
