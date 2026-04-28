import os
os.environ['TL_BACKEND'] = 'torch'
import json
import numpy as np
import scipy.sparse as sp

import gammagl.datasets as ggl_ds
import gammagl.transforms as ggl_t
import gammagl.utils as ggl_utils

from data_processor import *


class DataProcess_OGB(DataProcess):
    def __init__(self, name, path='../data/', rrz=0.5, seed=0) -> None:
        super().__init__(name, path=path, rrz=rrz, seed=seed)

    @property
    def n_train(self):
        if self.idx_train is None:
            self.input(['labels'])
        return len(self.idx_train)

    def fetch(self):
       
        dataset = ggl_ds.OGB(self.name, root='/share/data/dataset/OGB')
        split_idx = dataset.get_idx_split()
        idx_train, idx_val, idx_test = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph, labels = dataset[0]

        row, col = graph.edge_index[0].numpy(), graph.edge_index[1].numpy()
        row, col = np.concatenate([row, col], axis=0), np.concatenate([col, row], axis=0)
        deg = np.bincount(row)
        idx_zero = np.where(deg == 0)[0]
        if len(idx_zero) > 0:
            print(f"Warning: removing {len(idx_zero)} isolated nodes: {idx_zero}!")

        nodelst = deg.nonzero()[0]
        idxnew = np.full(graph.num_nodes, -1)
        idxnew[nodelst] = np.arange(len(nodelst))
        self._n = len(nodelst)
        row, col = idxnew[row], idxnew[col]
        self.adj_matrix = edgeidx2adj(row, col, self.n)
        self._m = self.adj_matrix.nnz

        self.attr_matrix = graph.x.numpy()[nodelst]
        assert (labels.ndim==2 and labels.shape[1]==1) or labels.ndim==1, "label shape error"
        self.labels = labels.numpy().flatten()[nodelst]

        
        self.idx_train = idxnew[idx_train]
        self.idx_train = self.idx_train[self.idx_train > -1]
        self.idx_val = idxnew[idx_val]
        self.idx_val = self.idx_val[self.idx_val > -1]
        self.idx_test = idxnew[idx_test]
        self.idx_test = self.idx_test[self.idx_test > -1]


class DataProcess_PyGFlickr(DataProcess):
    def __init__(self, name, path='../data/', rrz=0.5, seed=0) -> None:
        super().__init__(name, path=path, rrz=rrz, seed=seed)

    def fetch(self):
    
        root = '/share/data/dataset/PyG/Flickr/raw'
        f = np.load(os.path.join(root, 'adj_full.npz'))
        self.adj_matrix = sp.csr_matrix((f['data'], f['indices'], f['indptr']), f['shape'])
        self.to_undirected()
        self.attr_matrix = np.load(os.path.join(root, 'feats.npy'), allow_pickle=True)

        ys = [-1] * self.attr_matrix.shape[0]
        with open(os.path.join(root, 'class_map.json')) as f:
            class_map = json.load(f)
            for key, item in class_map.items():
                ys[int(key)] = item
        self.labels = np.array(ys).flatten()
        if self.labels.min() < 0:
            print(f"Warning: negative label: {self.labels.min()}")

        with open(os.path.join(root, 'role.json')) as f:
            role = json.load(f)
        self.idx_train = np.array(role['tr'])
        self.idx_val = np.array(role['va'])
        self.idx_test = np.array(role['te'])


class DataProcess_PyG(DataProcess):
    def __init__(self, name, path='../data/', rrz=0.5, seed=0) -> None:
        super().__init__(name, path=path, rrz=rrz, seed=seed)

    def fetch(self):
      
        dataset = ggl_ds.Coauthor(root='/share/data/dataset/PyG/Coauthor', name='CS', transform=ggl_t.ToUndirected())
        graph = dataset[0]
        degree = ggl_utils.degree(graph.edge_index[0], graph.num_nodes).numpy()
        idx_zero = np.where(degree == 0)[0]
        if len(idx_zero) > 0:
            print(f"Warning: removing {len(idx_zero)} isolated nodes: {idx_zero}!")

        self._n = graph.num_nodes
        self.adj_matrix = edgeidx2adj(graph.edge_index[0].numpy(), graph.edge_index[1].numpy(), self.n)
        self._m = self.adj_matrix.nnz
        self.attr_matrix = graph.x.numpy()
        self.labels = graph.y.numpy().flatten()
        if self.labels.min() < 0:
            print(f"Warning: negative label: {self.labels.min()}")


# ====================
if __name__ == '__main__':
   
    ds = DataProcess_OGB('ogbn-papers100M', path='/share/data/dataset/SCARA')
    # ds = DataProcess_PyGFlickr('flickr', path='/share/data/dataset/SCARA')
    # ds = DataProcess_PyG('cs', path='/share/data/dataset/SCARA')
    
    ds.fetch()
    ds.calculate(['deg', 'idx_train',])
    print(ds)
    os.makedirs(os.path.join(ds.path, ds.name), exist_ok=False)
   
    ds.output(['adjtxt', 'adjnpz', 'adjl', 'attr_matrix', 'deg', 'labels', 'attribute'])