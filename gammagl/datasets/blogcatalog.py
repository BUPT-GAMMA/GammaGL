import os
import os.path as osp
from typing import Callable, List, Optional
import numpy as np
import scipy.sparse as sp
import tensorlayerx as tlx
from gammagl.data import InMemoryDataset, download_url, Graph
import zipfile


class BlogCatalog(InMemoryDataset):
    r"""
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

        Tip
        ---
            .. list-table::
                :widths: 10 10 10 10
                :header-rows: 1

                * - #nodes
                  - #edges
                  - #features
                  - #classes
                * - 5,106
                  - 171,743
                  - 8,189
                  - 6
        """
    url = 'https://github.com/BUPT-GAMMA/SpCo/raw/main/dataset'

    def __init__(self, root: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.name = 'blog'
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = self.load_data(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> str:
        return self.name + '.zip'

    @property
    def processed_file_names(self) -> str:
        return tlx.BACKEND + 'data.pt'

    def download(self):
        path = download_url(f'{self.url}/{self.raw_file_names}', self.raw_dir)
        os.rename(path, osp.join(self.raw_dir, self.name + '.zip'))

    def process(self):
        zip_file = zipfile.ZipFile(file=osp.join(self.raw_dir, self.raw_file_names))
        if os.path.isdir(self.raw_dir):
            pass
        else:
            os.mkdir(self.raw_dir)
        for name in zip_file.namelist():
            zip_file.extract(name, self.raw_dir)
        zip_file.close()

        f_adj = np.load(file=osp.join(osp.join(self.raw_dir, self.name), 'adj.npz'))
        f_feat = sp.load_npz(file=osp.join(osp.join(self.raw_dir, self.name), 'feat.npz')).A
        f_label = np.load(file=osp.join(osp.join(self.raw_dir, self.name), 'label.npy'))

        adj = sp.csr_matrix((f_adj['data'], f_adj['indices'], f_adj['indptr']), f_adj['shape'])
        adj = adj.tocoo()
        row = adj.row
        col = adj.col
        edge_index = np.array([row, col], dtype=np.int64)

        feat = np.array(f_feat, dtype=np.float32)
        data = Graph(x=feat, edge_index=edge_index, y=f_label, to_tensor=True)
        node_index = list(range(data.num_nodes))
        np.random.shuffle(node_index)

        train_size, val_size = int(data.num_nodes * 0.5), int(data.num_nodes * 0.25)

        train_idx = node_index[0:train_size]
        val_idx = node_index[train_size:train_size + val_size]
        test_idx = node_index[train_size + val_size:]

        train_mask = tlx.zeros((data.num_nodes, 1)).squeeze(-1)
        val_mask = tlx.zeros((data.num_nodes, 1)).squeeze(-1)
        test_mask = tlx.zeros((data.num_nodes, 1)).squeeze(-1)

        train_mask[train_idx] = 1
        val_mask[val_idx] = 1
        test_mask[test_idx] = 1
        data.train_mask = train_mask.bool()
        data.val_mask = val_mask.bool()
        data.test_mask = test_mask.bool()

        data = data if self.pre_transform is None else self.pre_transform(data)

        self.save_data(self.collate([data]), self.processed_paths[0])
