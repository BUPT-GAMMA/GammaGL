import os
import os.path as osp
import zipfile
from typing import Optional, Callable, List
import numpy as np
import pandas as pd
import tensorlayerx as tlx
from gammagl.data import InMemoryDataset, download_url, Graph
from gammagl.utils import coalesce


class CustomDataset(InMemoryDataset):
    r"""统一加载四个额外数据集: roman-empire, tolokers, cora_full, ogbn-arxiv。
    继承自 GammaGL 的 InMemoryDataset,完全兼容 GammaGL 数据流水线。

    参数
    ----------
    root : str, optional
        数据集存储的根目录。默认为 './data'。
    name : str
        数据集名称，支持: 'roman-empire', 'tolokers', 'cora_full', 'ogbn-arxiv'。
    transform : callable, optional
        应用于每个 Graph 对象的变换函数（在访问时应用）。
    pre_transform : callable, optional
        在保存到磁盘前应用于 Graph 对象的变换函数。
    force_reload : bool, optional
        是否强制重新处理数据集。默认 False。

    Uniformly load four additional datasets: roman-empire, tolokers, cora_full, ogbn-arxiv.
    Inherits from GammaGL's InMemoryDataset, fully compatible with GammaGL data pipeline.

    Parameters
    ----------
    root : str, optional
        Root directory where the dataset is stored. Defaults to './data'.
    name : str
        Name of the dataset, supports: 'roman-empire', 'tolokers', 'cora_full', 'ogbn-arxiv'.
    transform : callable, optional
        A transform function to be applied to each Graph object (applied at access time).
    pre_transform : callable, optional
        A transform function to be applied to the Graph object before saving to disk.
    force_reload : bool, optional
        Whether to force reprocessing of the dataset. Defaults to False.
    """

    urls = {
        'roman-empire': 'https://github.com/yandex-research/heterophilous-graphs/raw/main/data/roman_empire.npz',
        'tolokers': 'https://github.com/yandex-research/heterophilous-graphs/raw/main/data/tolokers.npz',
        'cora_full': 'https://github.com/abojchevski/graph2gauss/raw/master/data/cora.npz',
        'ogbn-arxiv': 'http://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip',
    }

    def __init__(self, root: Optional[str] = None, name: str = 'roman-empire',
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 force_reload: bool = False):
        self.name = name.lower()
        assert self.name in self.urls, f"Unsupported dataset '{self.name}'. Choose from {list(self.urls.keys())}"

        if root is None:
            root = osp.join(osp.dirname(osp.abspath(__file__)), '..', 'data')
        self.root = root

        super().__init__(root, transform, pre_transform, force_reload=force_reload)
        self.data, self.slices = self.load_data(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        if self.name == 'ogbn-arxiv':
            return ['arxiv_loaded']
        else:
            return [f'{self.name}.npz']

    @property
    def processed_file_names(self) -> str:
        return tlx.BACKEND + '_data.pt'

    def download(self):
        url = self.urls[self.name]
        if self.name == 'ogbn-arxiv':
            zip_path = download_url(url, self.raw_dir, filename='arxiv.zip')
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(self.raw_dir)
            os.remove(zip_path)
            open(osp.join(self.raw_dir, 'arxiv_loaded'), 'a').close()
        else:
            download_url(url, self.raw_dir, filename=self.raw_file_names[0])

    def process(self):
        if self.name in ['roman-empire', 'tolokers']:
            data = self._process_hetero_npz()
        elif self.name == 'cora_full':
            data = self._process_cora_full()
        elif self.name == 'ogbn-arxiv':
            data = self._process_ogbn_arxiv()
        else:
            raise NotImplementedError(f"Processing for {self.name} not implemented.")
        if self.pre_transform is not None:
            data = self.pre_transform(data)
        self.save_data(self.collate([data]), self.processed_paths[0])

    def _process_hetero_npz(self) -> Graph:
        """处理 roman-empire 和 tolokers 数据集。"""
        path = osp.join(self.raw_dir, self.raw_file_names[0])
        with np.load(path, allow_pickle=False) as z:
            x = np.asarray(z['node_features'], dtype=np.float32)
            y = np.asarray(z['node_labels'], dtype=np.int64).reshape(-1)
            edges = np.asarray(z['edges'], dtype=np.int64)
        if edges.ndim == 2 and edges.shape[0] == 2:
            edge_index = edges
        elif edges.ndim == 2 and edges.shape[1] == 2:
            edge_index = edges.T
        else:
            raise ValueError(f'Unexpected edges shape in {self.name}.npz: {edges.shape}')
        edge_index = coalesce(edge_index)
        return Graph(x=x, edge_index=edge_index, y=y)

    def _process_cora_full(self) -> Graph:
        try:
            import scipy.sparse as sp
        except ImportError:
            raise ImportError("Loading 'cora_full' requires scipy. Install via: pip install scipy")
        path = osp.join(self.raw_dir, self.raw_file_names[0])
        with np.load(path, allow_pickle=False) as f:
            x = sp.csr_matrix(
                (f['attr_data'], f['attr_indices'], f['attr_indptr']),
                shape=tuple(f['attr_shape'])
            ).todense()
            x = np.asarray(x, dtype=np.float32)
            x[x > 0.0] = 1.0
            adj = sp.csr_matrix(
                (f['adj_data'], f['adj_indices'], f['adj_indptr']),
                shape=tuple(f['adj_shape'])
            ).tocoo()
            edge_index = np.stack([adj.row, adj.col], axis=0).astype(np.int64)
            y = np.asarray(f['labels'], dtype=np.int64).reshape(-1)
        edge_index = coalesce(edge_index)
        return Graph(x=x, edge_index=edge_index, y=y)

    def _process_ogbn_arxiv(self) -> Graph:
        arxiv_raw_dir = osp.join(self.raw_dir, 'arxiv', 'raw')
        split_dir = osp.join(self.raw_dir, 'arxiv', 'split', 'time')

        feat_path = osp.join(arxiv_raw_dir, 'node-feat.csv.gz')
        x = pd.read_csv(feat_path, compression='gzip', header=None).values.astype(np.float32)

        edge_path = osp.join(arxiv_raw_dir, 'edge.csv.gz')
        edge_index = pd.read_csv(edge_path, compression='gzip', header=None).values.T.astype(np.int64)

        label_path = osp.join(arxiv_raw_dir, 'node-label.csv.gz')
        y = pd.read_csv(label_path, compression='gzip', header=None).values.reshape(-1).astype(np.int64)

        train_idx = pd.read_csv(osp.join(split_dir, 'train.csv.gz'),
                            compression='gzip', header=None).values.reshape(-1)
        val_idx   = pd.read_csv(osp.join(split_dir, 'valid.csv.gz'),
                            compression='gzip', header=None).values.reshape(-1)
        test_idx  = pd.read_csv(osp.join(split_dir, 'test.csv.gz'),
                            compression='gzip', header=None).values.reshape(-1)

        num_nodes = x.shape[0]
        train_mask = np.zeros(num_nodes, dtype=bool)
        val_mask   = np.zeros(num_nodes, dtype=bool)
        test_mask  = np.zeros(num_nodes, dtype=bool)
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True

        edge_index = coalesce(edge_index)
        graph = Graph(x=x, edge_index=edge_index, y=y)
        graph.train_mask = train_mask
        graph.val_mask = val_mask
        graph.test_mask = test_mask
        return graph


    def get_idx_split(self):
        if self.name != 'ogbn-arxiv':
            raise NotImplementedError("Official split is only available for 'ogbn-arxiv'.")
        return {'train': self.train_idx, 'valid': self.val_idx, 'test': self.test_idx}

    def __repr__(self) -> str:
        return f"{self.name.capitalize()}()"