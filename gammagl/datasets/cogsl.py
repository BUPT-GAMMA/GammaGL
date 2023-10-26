import os
import os.path as osp
from typing import Callable, List, Optional
from gammagl.data import download_url,InMemoryDataset
import numpy as np
import scipy.sparse as sp
import tensorlayerx as tlx
from gammagl.data import Graph
from sklearn.metrics.pairwise import cosine_similarity as cos
from scipy.linalg import fractional_matrix_power, inv
import pickle


class CoGSLDataset(InMemoryDataset):
    r"""
        Three academic networks "Citeseer", "Wiki-CS" and "Academic", three non-graph
        datasets "Wine", "Breast Cancer" and "Digits" and a blog graph "Polblogs" from the
        `"Compact Graph Structure Learning via Mutual Information Compression"
        <https://arxiv.org/abs/2201.05540>`_ paper.

        Parameters
        ----------
            root: str
                Root directory where the dataset should be saved.
            name: str
                The name of the dataset (:obj:`"wine"`, :obj:`"breast_cancer"`,
                :obj:`"digits"`, :obj:`"polblogs"`, :obj:`"citeseer"`,
                :obj:`"wikics"`, :obj:`"ms"`).
            v1_p: int
                Hyper-parameter of view1 scope
            v2_p: int
                Hyper-parameter of view2 scope
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
                * - Wine
                  - 178
                  - 3560
                  - 13
                  - 3
                * - Cancer
                  - 569
                  - 22760
                  - 30
                  - 2
                * - Digits
                  - 1797
                  - 43128
                  - 64
                  - 10
                * - Polblogs
                  - 1222
                  - 33428
                  - 1490
                  - 2
                * - Citeseer
                  - 3327
                  - 9228
                  - 3703
                  - 6
                * - Wiki-CS
                  - 11701
                  - 291039
                  - 300
                  - 10
                * - MS Academic
                  - 18333
                  - 163788
                  - 6805
                  - 15
        """

    url = 'https://github.com/yuan429010/CoGSL/raw/main/dataset'

    def __init__(self, root: str = None, name:str = 'citeseer',
                 v1_p: int = 2, v2_p: int = 40,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.root = root
        self.name = name
        self.v1_p = v1_p
        self.v2_p = v2_p
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = self.load_data(self.processed_paths[0])
        self.view1 = sp.load_npz(self.raw_dir+'/view1.npz')
        self.view2 = sp.load_npz(self.raw_dir+'/view2.npz')
        with open(self.processed_dir+'/'+tlx.BACKEND+'_v1_indice_'+str(self.v1_p)+'.pt', 'rb') as f:
            self.v1_indice = tlx.convert_to_tensor(pickle.load(f), dtype=tlx.int64)
        with open(self.processed_dir+'/'+tlx.BACKEND+'_v2_indice_'+str(self.v2_p)+'.pt', 'rb') as f:
            self.v2_indice = tlx.convert_to_tensor(pickle.load(f), dtype=tlx.int64)


    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root ,self.name, 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        names = ['feat.npz', 'label.npy', 'test.npy', 'train.npy', 'val.npy', 'ori_adj.npz', 'view1.npz', 'view2.npz']
        return [f'{name}' for name in names]

    @property
    def processed_file_names(self) -> List[str]:
        names = ['data.pt']
        names.append(f'v1_indice_{self.v1_p}.pt')
        names.append(f'v2_indice_{self.v2_p}.pt')
        return [f'{tlx.BACKEND}_{name}' for name in names]

    def download(self):
        for fname in self.raw_file_names:
            download_url(f'{self.url}/{self.name}/{fname}', self.raw_dir)

    def process(self):
        feature = sp.load_npz(self.raw_dir + "/feat.npz")
        if self.name == "breast_cancer" or self.name == "digits" or self.name == "wine" or self.name == "wikics":
            feature = feature.todense()
        else:
            feature = preprocess_features(feature)
        x = tlx.convert_to_tensor(np.array(feature), dtype=tlx.float32)
        y = tlx.convert_to_tensor(np.load(self.raw_dir + "/label.npy"), dtype=tlx.int64)
        train_idx = np.load(self.raw_dir + "/train.npy")
        val_idx = np.load(self.raw_dir + "/val.npy")
        test_idx = np.load(self.raw_dir + "/test.npy")
        ori_adj = sp.load_npz(self.raw_dir + "/ori_adj.npz")
        ori_adj = ori_adj.tocoo()
        edge_index = tlx.convert_to_tensor([ori_adj.row, ori_adj.col])

        train_mask = index_to_mask(train_idx, size=y.shape[0])
        val_mask = index_to_mask(val_idx, size=y.shape[0])
        test_mask = index_to_mask(test_idx, size=y.shape[0])

        data = Graph(edge_index=edge_index, x=x, y=y)
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
        data.tensor()

        self.save_data(self.collate([data]), self.processed_paths[0])

        num_node = ori_adj.shape[0]

        view1 = sp.load_npz(self.raw_dir + '/view1.npz')
        v1_indice = get_khop_indices(self.v1_p, view1)

        view2 = sp.load_npz(self.raw_dir + '/view2.npz')
        if self.name =='wikics' or self.name =='ms':
            v2_indice = get_khop_indices(self.v1_p, view2)
        else:
            kn = topk(int(self.v2_p), view2)
            kn = sp.coo_matrix(kn)
            v2_indice = get_khop_indices(1, kn)

        with open(self.processed_dir+'/'+tlx.BACKEND+'_v1_indice_'+str(self.v1_p)+'.pt', 'wb') as f:
            pickle.dump(tlx.convert_to_numpy(v1_indice), f)
        with open(self.processed_dir+'/'+tlx.BACKEND+'_v2_indice_'+str(self.v2_p)+'.pt', 'wb') as f:
            pickle.dump(tlx.convert_to_numpy(v2_indice), f)


def index_to_mask(index, size):
    mask = np.zeros((size, ), dtype=bool)
    mask[index] = 1
    return mask


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()

def get_khop_indices(k, view):
    view = sp.csr_matrix((view.A > 0).astype("int32"))
    view_ = view
    for i in range(1, k):
        view_ = sp.csr_matrix((np.matmul(view_.toarray(), view.T.toarray()) > 0).astype("int32"))
    return tlx.convert_to_tensor(view_.nonzero())

def topk(k, adj):
    adj = adj.toarray()
    pos = np.zeros(adj.shape)
    for i in range(adj.shape[0]):
      one = adj[i].nonzero()[0]
      if len(one)>k:
        oo = np.argsort(-adj[i, one])
        sele = one[oo[:k]]
        pos[i, sele] = adj[i, sele]
      else:
        pos[i, one] = adj[i, one]
    return pos
