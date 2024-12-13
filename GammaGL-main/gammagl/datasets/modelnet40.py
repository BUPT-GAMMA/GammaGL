import numpy as np
import glob
import os
import os.path as osp
import shutil
from typing import Union, List, Tuple
import tensorlayerx as tlx
from gammagl.data import download_url, InMemoryDataset, extract_zip, Graph
import h5py


class ModelNet40(InMemoryDataset):
    r"""The ModelNet40 benchmark dataset used for classification from
    the `"Dynamic Graph CNN for Learning on Point Clouds"
    <https://arxiv.org/pdf/1801.07829.pdf>`_ paper, containing 12,311
    meshed CAD models from 40 categories.

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
    split: string, optional
        The type of dataset split
        (:obj:`"train"`, :obj:`"test"`).
    num_points: int, optional
        The number of points used to train or test.
    force_reload (bool, optional): Whether to re-process the dataset.
        (default: :obj:`False`)

    """
    url = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'

    def __init__(self, root=None, transform=None, pre_transform=None, pre_filter=None, split='train', num_points=1024, force_reload: bool = False):
        self.num_points = num_points
        self.split = split
        super(ModelNet40, self).__init__(root, transform, pre_transform, pre_filter, force_reload = force_reload)
        assert split in ['train', 'test']
        path = self.processed_paths[0] if split == 'train' else self.processed_paths[1]
        self.data, self.slices = self.load_data(path)

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return [f'ply_data_{split}*.h5'
                for split in ['train', 'test']]

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return [tlx.BACKEND + f'_{split}_data.pt'
                for split in ['train', 'test']]

    @property
    def num_classes(self) -> int:
        return super().num_classes()

    def download(self):
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        shutil.rmtree(self.raw_dir)
        name = self.url.split('/')[-1].split('.')[0]
        os.rename(osp.join(self.root, name), self.raw_dir)

    def process(self):
        for i, split in enumerate(['train', 'test']):
            data_list = []
            for h5_name in glob.glob(osp.join(self.raw_dir, f'ply_data_{split}*.h5')):
                f = h5py.File(h5_name)
                x = f['data'][:].astype('float32')
                y = f['label'][:].astype('int64')
                f.close()
                for j in range(x.shape[0]):
                    data_list.append(Graph(x=x[j][:self.num_points][:], y=y[j][0]))
            if self.pre_filter is not None and not self.pre_filter(data_list):
                data_list = self.pre_filter(data_list)
            if self.pre_transform is not None:
                data_list = self.pre_transform(data_list)
            print(self.collate(data_list))
            self.save_data(self.collate(data_list), self.processed_paths[i])
