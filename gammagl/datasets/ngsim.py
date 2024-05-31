import os
import os.path as osp
import zipfile
import tensorlayerx as tlx
from gammagl.data import Graph, HeteroGraph, download_url, extract_zip
from gammagl.data import InMemoryDataset
from typing import Callable, Optional, List


class NGSIM_US_101(InMemoryDataset):
    url = 'https://github.com/gjy1221/NGSIM-US-101/raw/main/data'

    def __init__(self, data_path, root: str = None, name: str = None,
                 transform: Optional[Callable] = None, pre_transform: Optional[Callable] = None,
                 force_reload: bool = False):
        self.name = name.lower()
        super().__init__(root, transform, pre_transform, force_reload=force_reload)
        self.data_path = data_path
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
        return osp.join(self.root, self.name)

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

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
            # 解压缩所有文件
            zip_ref.extractall(f'{self.root}/data')

