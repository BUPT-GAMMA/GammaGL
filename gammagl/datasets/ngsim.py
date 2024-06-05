import os
import os.path as osp
import zipfile
import tensorlayerx as tlx
from gammagl.data import Graph, HeteroGraph, download_url, extract_zip
from gammagl.data import InMemoryDataset
from typing import Callable, Optional, List


class NGSIM_US_101(InMemoryDataset):

    r"""
        The NGSIM US-101 dataset from the "NGSIM: Next Generation Simulation"
        <https://ops.fhwa.dot.gov/trafficanalysistools/ngsim.htm>`_ project,
        containing detailed vehicle trajectory data from the US-101 highway in
        Los Angeles, California.

        Parameters
        ----------
        root: str
            Root directory where the dataset should be saved.
        name: str, optional
            The name of the dataset (:obj:`"train", "val", "test"`).
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
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)

        """

    url = 'https://github.com/gjy1221/NGSIM-US-101/raw/main/data'

    def __init__(self, root: str = None, name: str = None,
                 transform: Optional[Callable] = None, pre_transform: Optional[Callable] = None,
                 force_reload: bool = False):
        self.name = name.lower()
        super().__init__(root, transform, pre_transform, force_reload=force_reload)
        self.data_path = f'{root}/processed/{name}'
        self.data_names = os.listdir('{}'.format(self.data_path))

    def __len__(self):
        return len(self.data_names)

    def __getitem__(self, index):
        data_item = tlx.files.load_npy_to_any(self.data_path, self.data_names[index])
        data_item.edge_attr = data_item.edge_attr.transpose(0, 1)
        data_item.edge_type = data_item.edge_type.transpose(0, 1)
        return data_item

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, 'raw', self.name)

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        return [f'/{self.name.lower()}.zip']

    @property
    def processed_file_names(self) -> str:
        return tlx.BACKEND + '_data.pt'

    def download(self):
        print(self.root)
        path = download_url(self.url + self.raw_file_names[0], self.raw_dir)
        with zipfile.ZipFile(path, 'r') as zip_ref:
            zip_ref.extractall(self.processed_dir)

