import os.path as osp
from typing import Callable, Optional
import tensorlayerx as tlx
from gammagl.data import InMemoryDataset, download_url
from gammagl.io.npz import read_npz


class Coauthor(InMemoryDataset):
    r"""The Coauthor CS and Coauthor Physics networks from the
    `"Pitfalls of Graph Neural Network Evaluation"
    <https://arxiv.org/abs/1811.05868>`_ paper.
    Nodes represent authors that are connected by an edge if they co-authored a
    paper.
    Given paper keywords for each author's papers, the task is to map authors
    to their respective field of study.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"CS"`,
            :obj:`"Physics"`).
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
            * - CS
              - 18,333
              - 163,788
              - 6,805
              - 15
            * - Physics
              - 34,493
              - 495,924
              - 8,415
              - 5
    """

    url = 'https://github.com/shchur/gnn-benchmark/raw/master/data/npz/'

    def __init__(self, root: str = None, name: str = 'cs',
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        assert name.lower() in ['cs', 'physics']
        self.name = 'CS' if name.lower() == 'cs' else 'Physics'
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = self.load_data(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> str:
        return f'ms_academic_{self.name[:3].lower()}.npz'

    @property
    def processed_file_names(self) -> str:
        return tlx.BACKEND + '_data.pt'

    def download(self):
        download_url(self.url + self.raw_file_names, self.raw_dir)

    def process(self):
        data = read_npz(self.raw_paths[0])
        data = data if self.pre_transform is None else self.pre_transform(data)
        self.save_data(self.collate([data]), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}{self.name}()'

# data=Coauthor(root='./Coauthor/',name='cs')
# data.process()
