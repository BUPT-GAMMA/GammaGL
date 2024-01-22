import os
import os.path as osp
from gammagl.data import (InMemoryDataset, download_url,
                          extract_zip)
import numpy as np
from gammagl.data import Graph
import pandas as pd


class MLDataset(InMemoryDataset):
    # url = 'https://recbole.s3-accelerate.amazonaws.com/ProcessedDatasets/MovieLens/ml-100k.zip'

    def __init__(self, root=None, split='train', transform=None, pre_transform=None,
                 pre_filter=None, dataset_name='ml-100k', force_reload: bool = False):
        assert split in ['train', 'val', 'valid', 'test']

        assert dataset_name in ['ml-100k', 'ml-1m', 'ml-10m', 'ml-20m']
        self.dataset_name = dataset_name

        url_pre = 'https://recbole.s3-accelerate.amazonaws.com/ProcessedDatasets/MovieLens/'
        url_post = '.zip'
        self.url = f'{url_pre}{self.dataset_name}{url_post}'

        super().__init__(root, transform, pre_transform, pre_filter, force_reload = force_reload)

        self.data, self.slices = self.load_data(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, 'ml', 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'ml', 'processed')

    @property
    def raw_file_names(self):
        return [f'{self.dataset_name}.user', f'{self.dataset_name}.item', f'{self.dataset_name}.inter']

    @property
    def processed_file_names(self):
        return [f'{self.dataset_name}.pt']

    def download(self):
        path = download_url(self.url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.remove(path)

    def process(self):
        path = osp.join(self.raw_dir, f'{self.dataset_name}.inter')

        inter_df = pd.read_csv(path, delimiter='\t')

        edge_index = np.array([inter_df['user_id:token'].to_numpy() - 1, inter_df['item_id:token'].to_numpy() - 1])
        edge_weight = inter_df['rating:float'].to_numpy()

        path = osp.join(self.raw_dir, f'{self.dataset_name}.user')
        user_df = pd.read_csv(path, delimiter='\t')
        user_id = user_df['user_id:token'].to_numpy() - 1

        path = osp.join(self.raw_dir, f'{self.dataset_name}.item')
        item_df = pd.read_csv(path, delimiter='\t')
        item_id = item_df['item_id:token'].to_numpy() - 1

        graph = Graph(edge_index=edge_index, edge_weight=edge_weight, user_id=user_id, item_id=item_id)
        self.save_data(self.collate([graph]), self.processed_paths[0])
