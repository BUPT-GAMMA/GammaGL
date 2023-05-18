import numpy as np
import tensorlayerx as tlx
import os.path as osp
from typing import Callable, List, Optional
from gammagl.data import download_url
from gammagl.data import InMemoryDataset
from gammagl.io.planetiod import read_planetoid_data

try:
    import cPickle as pickle
except ImportError:
    import pickle


class Planetoid(InMemoryDataset):
    r"""
    The citation network datasets "Cora", "CiteSeer" and "PubMed" from the
    `"Revisiting Semi-Supervised Learning with Graph Embeddings"
    <https://arxiv.org/abs/1603.08861>`_ paper.
    Nodes represent documents and edges represent citation links.
    Training, validation and test splits are given by binary masks.
    
    Parameters
    ----------
        root: str
            Root directory where the dataset should be saved.
        name: str
            The name of the dataset (:obj:`"Cora"`,
            :obj:`"CiteSeer"`, :obj:`"PubMed"`).
        split: string
            The type of dataset split
            (:obj:`"public"`, :obj:`"full"`, :obj:`"random"`).
            If set to :obj:`"public"`, the split will be the public fixed split
            from the
            `"Revisiting Semi-Supervised Learning with Graph Embeddings"
            <https://arxiv.org/abs/1603.08861>`_ paper.
            If set to :obj:`"full"`, all nodes except those in the validation
            and test sets will be used for training (as in the
            `"FastGCN: Fast Learning with Graph Convolutional Networks via
            Importance Sampling" <https://arxiv.org/abs/1801.10247>`_ paper).
            If set to :obj:`"random"`, train, validation, and test sets will be
            randomly generated, according to :obj:`num_train_per_class`,
            :obj:`num_val` and :obj:`num_test`. (default: :obj:`"public"`)
        num_train_per_class: int, optional
            The number of training samples
            per class in case of :obj:`"random"` split. (default: :obj:`20`)
        num_val: int, optional
            The number of validation samples in case of
            :obj:`"random"` split. (default: :obj:`500`)
        num_test: int, optional
            The number of test samples in case of
            :obj:`"random"` split. (default: :obj:`1000`)
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

    Tip
    ---
        .. list-table::
            :widths: 10 10 10 10 10
            :header-rows: 1
            
            * - Name
              - #nodes
              - #edges
              - #features
              - #classes
            * - Cora
              - 2,708
              - 10,556
              - 1,433
              - 7
            * - CiteSeer
              - 3,327
              - 9,104
              - 3,703
              - 6
            * - PubMed
              - 19,717
              - 88,648
              - 500
              - 3
    """

    url = 'https://github.com/kimiyoung/planetoid/raw/master/data'

    def __init__(self, root: str = None, name: str = 'cora', split: str = "public",
                 num_train_per_class: int = 20, num_val: int = 500,
                 num_test: int = 1000, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.name = name

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = self.load_data(self.processed_paths[0])
        self.split = split
        assert self.split in ['public', 'full', 'random']

        if split == 'full':
            data = self.get(0)
            data.numpy()
            data.train_mask.fill(True)
            data.train_mask[data.val_mask | data.test_mask] = False
            data.tensor()
            self.data, self.slices = self.collate([data])

        elif split == 'random':
            data = self.get(0)
            data.numpy()
            data.train_mask.fill(False)
            for c in range(self.num_classes):
                idx = np.array((data.y == c).nonzero()).reshape((-1))
                idx = idx[np.random.permutation(idx.shape[0])[:num_train_per_class]]
                data.train_mask[idx] = True

            remaining = np.array((~data.train_mask).nonzero()).reshape((-1))
            # remaining = (~data.train_mask).nonzero(as_tuple=False).view(-1)
            remaining = remaining[np.random.permutation(remaining.shape[0])]

            data.val_mask.fill(False)
            data.val_mask[remaining[:num_val]] = True

            data.test_mask.fill(False)
            data.test_mask[remaining[num_val:num_val + num_test]] = True
            data.tensor()
            self.data, self.slices = self.collate([data])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        names = ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']
        return [f'ind.{self.name.lower()}.{name}' for name in names]

    @property
    def processed_file_names(self) -> str:
        return tlx.BACKEND + '_data.pt'

    def download(self):
        for name in self.raw_file_names:
            download_url(f'{self.url}/{name}', self.raw_dir)

    def process(self):
        data = read_planetoid_data(self.raw_dir, self.name)
        data = data if self.pre_transform is None else self.pre_transform(data)
        self.save_data(self.collate([data]), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name}()'
