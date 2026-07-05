import os
import numpy as np
import tensorlayerx as tlx
from gammagl.datasets import Planetoid, WebKB
from gammagl.utils import  to_undirected, remove_self_loops
from gammagl.datasets.custom_datasets import CustomDataset

script_dir = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = script_dir + "/Datasets"

class NodeDataset:
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name.lower()
        self.file_root = DATASET_ROOT

        if self.dataset_name in ['pubmed']:
            dataset = Planetoid(root=self.file_root, name=self.dataset_name, force_reload=True)
            data = dataset[0]
        elif self.dataset_name in ['cornell', 'texas', 'wisconsin']:
            dataset = WebKB(root=self.file_root, name=self.dataset_name, force_reload=True)
            data = dataset[0]
        else:
            data = self._load_custom_dataset()

        self.x = tlx.convert_to_tensor(data.x, dtype=tlx.float32)
        self.edge_index = tlx.convert_to_tensor(data.edge_index, dtype=tlx.int64)
        self.y = tlx.convert_to_tensor(data.y, dtype=tlx.int64)

        if self.dataset_name == 'ogbn-arxiv' and len(self.y.shape) > 1:
            self.y = tlx.reshape(self.y, (-1,))

        self.nfeat = self.x.shape[1]
        self.nclass = int(tlx.reduce_max(self.y)) + 1
        self.nnode = self.x.shape[0]

        self.edge_index = remove_self_loops(self.edge_index)[0]
        self.edge_index = to_undirected(self.edge_index)

        self._normalize_features()

    def _load_custom_dataset(self):
        if self.dataset_name not in ['roman-empire', 'tolokers', 'cora_full', 'ogbn-arxiv']:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
        dataset = CustomDataset(root=self.file_root, name=self.dataset_name)
        return dataset[0]

    def _normalize_features(self):
        self.x = tlx.where(tlx.is_nan(self.x), tlx.zeros_like(self.x), self.x)
        rowsum = tlx.reduce_sum(self.x, axis=1, keepdims=True)
        rowsum = tlx.where(rowsum == 0., tlx.ones_like(rowsum), rowsum)
        self.x = self.x / rowsum

    def random_split(self, seed: int, p_train: float = 0.6, p_val: float = 0.2):
        tlx.set_seed(seed)
        np.random.seed(seed)
        n_train = int(self.nnode * p_train)
        n_val = int(self.nnode * p_val)
        full_idx = np.random.permutation(self.nnode)
        train_idx = full_idx[:n_train]
        val_idx = full_idx[n_train+1:n_train + n_val]
        test_idx = full_idx[n_train + n_val+1:]
        return train_idx, val_idx, test_idx
