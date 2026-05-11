import os as _os

_os.environ.setdefault('TL_BACKEND', 'torch')

import tensorlayerx as tlx

if getattr(tlx, 'BACKEND', 'backend') == 'backend':
    tlx.BACKEND = 'torch'

import torch
import numpy as np
from gammagl.data import Graph, InMemoryDataset, download_url
from gammagl.utils.get_laplacian import get_laplacian
from gammagl.datasets import (
    Amazon, Coauthor, Planetoid, Reddit, Flickr,
    PolBlogs, WikiCS, TUDataset
)
try:
    from ogb.nodeproppred import PygNodePropPredDataset
    from torch_geometric.transforms import RandomNodeSplit
    OGB_AVAILABLE = True
except ImportError:
    OGB_AVAILABLE = False
from scipy.sparse import csr_matrix, diags, eye
from scipy.sparse.linalg import eigs, eigsh
import os

input_dim_dict = {
    "KarateClub": 34, "Cora": 1433, "Citeseer": 3703, "PubMed": 500,
    'ogbn-arxiv': 128, "CS": 6805, "GitHub": 128, "USA": 1190, "computers": 767,
    "Flickr": 500, "WikiCS": 128, "COLLAB": 128, "Photo": 745, "Europe": 399, "Brazil": 131
}
class_num_dict = {
    "KarateClub": 4, "Cora": 7, "Citeseer": 6, "PubMed": 3, "ogbn-arxiv": 40, "CS": 15,
    "GitHub": 2, "USA": 4, "computers": 10, "Flickr": 7, "WikiCS": 10, "COLLAB": 3,
    "Photo": 8, "Europe": 4, "Brazil": 4
}


def add_train_val_test_mask(data, num_val=0.1, num_test=0.2, num_train_per_class=None):
    num_classes = int(tlx.reduce_max(data.y)) + 1
    num_nodes = data.num_nodes
    y_np = tlx.convert_to_numpy(data.y)

    train_mask = np.zeros(num_nodes, dtype=np.bool_)
    val_mask = np.zeros(num_nodes, dtype=np.bool_)
    test_mask = np.zeros(num_nodes, dtype=np.bool_)

    if num_train_per_class is not None:
        for c in range(num_classes):
            idx = np.where(y_np == c)[0]
            perm = torch.randperm(len(idx)).cpu().numpy()
            idx = idx[perm[:num_train_per_class]]
            train_mask[idx] = True
    else:
        perm = torch.randperm(num_nodes).cpu().numpy()
        num_test_nodes = int(num_test * num_nodes)
        num_val_nodes = int(num_val * num_nodes)
        num_train_nodes = max(0, num_nodes - num_test_nodes - num_val_nodes)
        train_mask[perm[:num_train_nodes]] = True
        val_mask[perm[num_train_nodes:num_train_nodes + num_val_nodes]] = True
        test_mask[perm[num_train_nodes + num_val_nodes:]] = True
        data.train_mask = tlx.convert_to_tensor(train_mask)
        data.val_mask = tlx.convert_to_tensor(val_mask)
        data.test_mask = tlx.convert_to_tensor(test_mask)
        return data

    remaining = np.where(~train_mask)[0]
    perm_remaining = torch.randperm(len(remaining)).cpu().numpy()
    remaining = remaining[perm_remaining]

    num_test_nodes = int(num_test * num_nodes)
    num_val_nodes = int(num_val * num_nodes)

    test_mask[remaining[:num_test_nodes]] = True
    val_mask[remaining[num_test_nodes:num_test_nodes + num_val_nodes]] = True

    train_mask = train_mask & (~val_mask) & (~test_mask)

    data.train_mask = tlx.convert_to_tensor(train_mask)
    data.val_mask = tlx.convert_to_tensor(val_mask)
    data.test_mask = tlx.convert_to_tensor(test_mask)

    return data


def load_data(root: str, data_name: str,
              num_val=0.1, num_test=0.2, num_per_class=None):
    if data_name in ["computers", "Photo"]:
        dataset = Amazon(root, name=data_name)
        dataset.transform = lambda g: add_train_val_test_mask(
            g, num_val=num_val, num_test=num_test, num_train_per_class=num_per_class
        )
    elif data_name == "KarateClub":
        dataset = KarateClubDataset(root=root, transform=lambda data: add_train_val_test_mask(data, num_val=0.2, num_test=0.3))
    elif data_name in ["CS", "Physics"]:
        dataset = Coauthor(root, name=data_name)
        dataset.transform = lambda g: add_train_val_test_mask(
            g, num_val=0.2, num_test=0.3, num_train_per_class=num_per_class
        )
    elif data_name in ['Cora', 'Citeseer', 'PubMed']:
        if num_per_class is None:
            num_per_class = 20
            split = "public"
        else:
            split = "random"
        dataset = Planetoid(root, name=data_name, split=split, num_train_per_class=num_per_class)
    elif data_name == 'ogbn-arxiv':
        if not OGB_AVAILABLE:
            raise ImportError("ogbn-arxiv requires OGB library. Please install with: pip install ogb torch-geometric")
        dataset = PygNodePropPredDataset(name=data_name, root=root,
                                         transform=RandomNodeSplit(num_val=0.2, num_test=0.3))
    elif data_name == 'GitHub':
        dataset = GitHubDataset(root=root, transform=RandomNodeSplit(num_val=0.1, num_test=0.2))
    elif data_name in ["USA", "Brazil", "Europe"]:
        dataset = AirportsDataset(root=root, name=data_name, transform=RandomNodeSplit(num_val=0.1, num_test=0.2))
    elif data_name == 'Flickr':
        dataset = Flickr(os.path.join(root, data_name))
    elif data_name == 'Reddit':
        dataset = Reddit(root)
    elif data_name == 'PolBlogs':
        dataset = PolBlogs(root, transform=RandomNodeSplit(num_val=0.1, num_test=0.2))
    elif data_name == "WikiCS":
        dataset = WikiCS(os.path.join(root, data_name), transform=RandomNodeSplit(num_val=0.1, num_test=0.2))
    elif data_name in ["COLLAB"]:
        dataset = TUDataset(root, data_name)
    else:
        raise NotImplementedError(f"Dataset {data_name} not implemented")

    return dataset


def get_eigen_tokens(data, embed_dim, device=None):
    n = data.num_nodes
    if n is None:
        if hasattr(data, 'x') and data.x is not None:
            n = int(data.x.shape[0])
        else:
            if hasattr(data.edge_index, 'cpu'):
                edge_index_np = np.array(data.edge_index.cpu().numpy())
            elif hasattr(data.edge_index, 'numpy'):
                edge_index_np = np.array(data.edge_index.numpy())
            else:
                edge_index_np = np.array(data.edge_index)
            n = int(np.max(edge_index_np)) + 1
    if hasattr(n, 'item'):
        n = int(n.item())
    n = int(n)

    with torch.device('cpu'):
        edge_index_cpu = data.edge_index.cpu() if hasattr(data.edge_index, 'cpu') else data.edge_index
        edge_index, edge_weight = get_laplacian(edge_index_cpu, normalization="sym", num_nodes=n)

    if hasattr(edge_index, 'detach'):
        edge_index = edge_index.detach().cpu().numpy()
    elif hasattr(edge_index, 'numpy'):
        edge_index = edge_index.numpy()
    edge_index = np.asarray(edge_index)
    if edge_index.ndim != 2:
        edge_index = edge_index.reshape(2, -1)
    if edge_index.shape[0] != 2 and edge_index.shape[1] == 2:
        edge_index = edge_index.T

    if hasattr(edge_weight, 'detach'):
        edge_weight = edge_weight.detach().cpu().numpy()
    elif hasattr(edge_weight, 'numpy'):
        edge_weight = edge_weight.numpy()
    edge_weight = np.asarray(edge_weight).reshape(-1)

    row = edge_index[0].reshape(-1).astype(np.int64)
    col = edge_index[1].reshape(-1).astype(np.int64)
    m = min(len(row), len(col), len(edge_weight))
    row, col, edge_weight = row[:m], col[:m], edge_weight[:m]

    valid = (row >= 0) & (row < n) & (col >= 0) & (col < n)
    row = row[valid]
    col = col[valid]
    edge_weight = edge_weight[valid]
    L = csr_matrix((edge_weight, (row, col)), shape=(n, n))

    k = min(embed_dim, L.shape[0] - 2)
    if k <= 0:
        eigvecs = np.zeros((n, embed_dim), dtype=np.float32)
    else:
        try:
            _, eigvecs = eigs(
                L,
                k=k,
                which='SM',
                ncv=min(20 + 2 * k, L.shape[0]),
                tol=1e-3,
                maxiter=1000,
            )
            eigvecs = np.asarray(eigvecs.real, dtype=np.float32)
        except Exception:
            eigvecs = np.zeros((n, k), dtype=np.float32)

        if k < embed_dim:
            padding = np.zeros((eigvecs.shape[0], embed_dim - k + 2), dtype=eigvecs.dtype)
            eigvecs = np.concatenate([eigvecs, padding], axis=-1)

    def tokens(idx):
        if hasattr(idx, 'detach'):
            idx = idx.detach().cpu().numpy()
        elif hasattr(idx, 'cpu'):
            idx = idx.cpu().numpy()
        elif hasattr(idx, 'numpy'):
            idx = idx.numpy()
        elif hasattr(idx, 'tolist'):
            idx = idx.tolist()
        if isinstance(idx, int):
            result = eigvecs[idx]
        else:
            idx = [int(i) for i in idx]
            result = eigvecs[idx]
        return tlx.convert_to_tensor(np.array(result), dtype=tlx.float32).cpu()

    return tokens


class KarateClubDataset(InMemoryDataset):
    def __init__(self, root=None, transform=None, pre_transform=None, force_reload=False):
        super().__init__(root, transform, pre_transform, force_reload=force_reload)
        self.data, self.slices = self.load_data(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return tlx.BACKEND + '_data.pt'

    @property
    def raw_file_names(self):
        return []

    def download(self):
        pass

    def process(self):
        row = [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4,
            5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 10, 10,
            10, 11, 12, 12, 13, 13, 13, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17,
            18, 18, 19, 19, 19, 20, 20, 21, 21, 22, 22, 23, 23, 23, 23, 23, 24,
            24, 24, 25, 25, 25, 26, 26, 27, 27, 27, 27, 28, 28, 28, 29, 29, 29,
            29, 30, 30, 30, 30, 31, 31, 31, 31, 31, 31, 32, 32, 32, 32, 32, 32,
            32, 32, 32, 32, 32, 32, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33,
            33, 33, 33, 33, 33, 33
        ]
        col = [
            1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 17, 19, 21, 31, 0, 2, 3, 7,
            13, 17, 19, 21, 30, 0, 1, 3, 7, 8, 9, 13, 27, 28, 32, 0, 1, 2, 7,
            12, 13, 0, 6, 10, 0, 6, 10, 16, 0, 4, 5, 16, 0, 1, 2, 3, 0, 2, 30,
            32, 33, 2, 33, 0, 4, 5, 0, 0, 3, 0, 1, 2, 3, 33, 32, 33, 32, 33, 5,
            6, 0, 1, 32, 33, 0, 1, 33, 32, 33, 0, 1, 32, 33, 25, 27, 29, 32,
            33, 25, 27, 31, 23, 24, 31, 29, 33, 2, 23, 24, 33, 2, 31, 33, 23,
            26, 32, 33, 1, 8, 32, 33, 0, 24, 25, 28, 32, 33, 2, 8, 14, 15, 18,
            20, 22, 23, 29, 30, 31, 33, 8, 9, 13, 14, 15, 18, 19, 20, 22, 23,
            26, 27, 28, 29, 30, 31, 32
        ]

        num_nodes = 34
        edge_index = np.array([row, col], dtype=np.int64)

        x = np.eye(num_nodes, dtype=np.float32)

        y = np.array([
            1, 1, 1, 1, 3, 3, 3, 1, 0, 1, 3, 1, 1, 1, 0, 0, 3, 1, 0, 1, 0, 1,
            0, 0, 2, 2, 0, 0, 2, 0, 0, 2, 0, 0
        ], dtype=np.int64)

        train_mask = np.zeros(num_nodes, dtype=np.bool_)
        for i in range(int(y.max()) + 1):
            idx = np.where(y == i)[0][0]
            train_mask[idx] = True

        data = Graph(x=x, edge_index=edge_index, y=y, num_nodes=num_nodes)
        data.train_mask = tlx.convert_to_tensor(train_mask)
        data.val_mask = tlx.zeros((num_nodes,), dtype=tlx.bool)
        data.test_mask = tlx.zeros((num_nodes,), dtype=tlx.bool)

        data = data if self.pre_transform is None else self.pre_transform(data)
        self.save_data(self.collate([data]), self.processed_paths[0])


class GitHubDataset(InMemoryDataset):
    url = 'https://raw.githubusercontent.com/EdisonLeeeee/GraphData/master/datasets/git_web_sp.npz'

    def __init__(self, root=None, transform=None, pre_transform=None, force_reload=False):
        super().__init__(root, transform, pre_transform, force_reload=force_reload)
        self.data, self.slices = self.load_data(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return 'git_web_sp.npz'

    @property
    def processed_file_names(self):
        return tlx.BACKEND + '_data.pt'

    def download(self):
        download_url(self.url, self.raw_dir)

    def process(self):
        data = np.load(self.raw_paths[0], allow_pickle=True)

        x = data['features'].astype(np.float32)
        y = data['target'].astype(np.int64)
        edge_index = data['edges'].astype(np.int64)
        edge_index = edge_index.T

        graph = Graph(x=x, y=y, edge_index=edge_index)
        graph = add_train_val_test_mask(graph, num_val=0.1, num_test=0.2)

        graph = graph if self.pre_transform is None else self.pre_transform(graph)
        self.save_data(self.collate([graph]), self.processed_paths[0])


class AirportsDataset(InMemoryDataset):
    edge_url = 'https://raw.githubusercontent.com/leoribeiro/struc2vec/master/graph/{}-airports.edgelist'
    label_url = 'https://raw.githubusercontent.com/leoribeiro/struc2vec/master/graph/labels-{}-airports.txt'

    def __init__(self, root=None, name='usa', transform=None, pre_transform=None, force_reload=False):
        self.name = name.lower()
        assert self.name in ['usa', 'brazil', 'europe']
        super().__init__(root, transform, pre_transform, force_reload=force_reload)
        self.data, self.slices = self.load_data(self.processed_paths[0])

    @property
    def raw_dir(self):
        return os.path.join(self.root, self.name.capitalize(), 'raw')

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.name.capitalize(), 'processed')

    @property
    def raw_file_names(self):
        return [
            f'{self.name}-airports.edgelist',
            f'labels-{self.name}-airports.txt',
        ]

    @property
    def processed_file_names(self):
        return tlx.BACKEND + '_data.pt'

    def download(self):
        download_url(self.edge_url.format(self.name), self.raw_dir)
        download_url(self.label_url.format(self.name), self.raw_dir)

    def process(self):
        index_map, ys = {}, []
        with open(self.raw_paths[1], 'r') as f:
            rows = f.read().split('\n')[1:-1]
            for i, row in enumerate(rows):
                idx, label = row.split()
                index_map[int(idx)] = i
                ys.append(int(label))

        num_nodes = len(ys)
        y = np.array(ys, dtype=np.int64)
        x = np.eye(num_nodes, dtype=np.float32)

        edge_indices = []
        with open(self.raw_paths[0], 'r') as f:
            rows = f.read().split('\n')[:-1]
            for row in rows:
                src, dst = row.split()
                edge_indices.append([index_map[int(src)], index_map[int(dst)]])

        edge_index = np.array(edge_indices, dtype=np.int64).T

        graph = Graph(x=x, y=y, edge_index=edge_index)
        graph = add_train_val_test_mask(graph, num_val=0.1, num_test=0.2)

        graph = graph if self.pre_transform is None else self.pre_transform(graph)
        self.save_data(self.collate([graph]), self.processed_paths[0])
