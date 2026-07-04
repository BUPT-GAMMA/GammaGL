import os
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Optional, List


NTRAIN_PER_CLASS = 20
NVAL_PER_CLASS = NTRAIN_PER_CLASS * 10


# ====================
def diag_sp(diag) -> sp.dia_matrix:
    """Diagonal array to scipy sparse diagonal matrix"""
    n = len(diag)
    return sp.dia_matrix((diag, [0]), shape=(n, n))


def to_torch_sparse(x: sp.coo_matrix):
    import torch
    x = x.tocoo()
    shape = x.shape
    i = torch.LongTensor(np.vstack((x.row, x.col)))
    v = torch.FloatTensor(x.data)
    return torch.sparse.FloatTensor(i, v, torch.Size(shape))


def matstd(m, with_mean=False):
    """Matrix standardization"""
    scaler = StandardScaler(with_mean=with_mean)
    m = scaler.fit_transform(m)
    return m


def matnorm_inf_dual(m, axis=0):
    """Normalization of matrix, set positive/negative sum of column to 1
    """
    pos = m.clip(min=0)
    possum = pos.sum(axis=axis)
    possum[possum == 0] = 1         # Avoid sum = 0
    pos = pos / possum

    neg = m.clip(max=0)
    negsum = - neg.sum(axis=axis)
    negsum[negsum == 0] = 1         # Avoid sum = 0
    neg = neg / negsum
    return (pos + neg)


def matstd_clip(m, idx, with_mean=False, clip=False):
    """Standardize and clip per feature"""
    # idx = np.setdiff1d(idx, [0])
    if (len(idx) > 0.75 * m.shape[0]) and (m.shape[0] > 2e9):
        idx = np.random.choice(idx, size=int(len(idx)/5), replace=False)
    scaler = StandardScaler(with_mean=with_mean)
    scaler.fit(m[idx])
    if clip:
        mean, std = scaler.mean_, scaler.scale_
        k = 9
        m = np.clip(m, a_min=mean-k*std, a_max=mean+k*std)
    m = scaler.transform(m)
    return m


def edgeidx2adj(row, col, n, undirected=False):
    if undirected:
        row, col = np.concatenate([row, col], axis=0), np.concatenate([col, row], axis=0)
    ones = np.ones(len(row), dtype=np.int8)
    adj = sp.coo_matrix(
            (ones, (row, col)),
            shape=(n, n))
    adj = adj.tocsr()
    adj.setdiag(0)
    adj.eliminate_zeros()
    adj.data = np.ones(len(adj.data), dtype=np.int8)
    return adj


def split_random(seed, n, n_train, n_val):
    """Split index randomly"""
    np.random.seed(seed)
    rnd = np.random.permutation(n)

    train_idx = np.sort(rnd[:n_train])
    val_idx = np.sort(rnd[n_train:n_train + n_val])

    train_val_idx = np.concatenate((train_idx, val_idx))
    test_idx = np.sort(np.setdiff1d(np.arange(n), train_val_idx))
    return train_idx, val_idx, test_idx


def split_label(seed, n, n_train_per_class, n_val, labels):
    """Split index with equal label in train set"""
    np.random.seed(seed)
    rnd = set(np.arange(n))

    train_idx = np.array([], dtype=int)
    if labels.ndim == 1:
        lb_nonnan = labels[~np.isnan(labels)]
        nclass = int(lb_nonnan.max()) + 1
        for i in range(nclass):
            cdd = np.where(labels == i)[0]
            sz = min(n_train_per_class, len(cdd))
            idxi = np.random.choice(cdd, size=sz, replace=False)
            train_idx = np.concatenate((train_idx, idxi))
    else:
        nclass = labels.shape[1]
        for i in range(nclass):
            cdd = np.where(labels[:, i] > 0)[0]
            sz = min(n_train_per_class, len(cdd))
            idxi = np.random.choice(cdd, size=sz, replace=False)
            train_idx = np.concatenate((train_idx, idxi))

    train_idx = np.unique(train_idx.flatten(), axis=0)
    val_idx = np.array((list( rnd - set(train_idx) )))
    val_idx = np.random.choice(val_idx, size=n_val, replace=False)
    val_idx = np.sort(val_idx)

    train_val_idx = np.concatenate((train_idx, val_idx))
    test_idx = np.sort(np.setdiff1d(np.arange(n), train_val_idx))
    return train_idx, val_idx, test_idx


def split_stratify(seed, n, n_train, n_val, labels):
    assert labels.ndim == 1, 'Only support 1D labels'
    idx = np.arange(n)
    train_idx, test_idx = train_test_split(idx, train_size=n_train, random_state=seed, stratify=labels)
    val_idx, test_idx = train_test_split(test_idx, train_size=n_val, random_state=seed, stratify=labels[test_idx])
    return train_idx, val_idx, test_idx


# ====================
class DataProcess(object):
    def __init__(self, name: str, path: str='../data/', rrz: float=0.5, seed: int=0) -> None:
        super().__init__()
        self.name = name
        self.path = path
        self.rrz = rrz
        self.seed = seed

        self._n = None
        self._m = None
        self._nfeat = None
        self._nclass = None

        self.adjnpz_path = self._get_path('adj.npz')
        self.adjtxt_path = self._get_path('adj.txt')
        self.degree_path = self._get_path('degree.npz')
        self.labels_path = self._get_path('labels.npz')
        self.query_path = self._get_path('query.txt')
        self.querytrain_path = self._get_path('query_train.txt')
        self.feats_path = self._get_path('feats.npy')
        self.featsnorm_path = self._get_path(f'feats_normt_{self.rrz:g}.npy')

        self.adj_matrix = None
        self.deg = None
        self.labels = None              # Labels can be 1D array or 2D one hot
        self.idx_train = None
        self.idx_val = None
        self.idx_test = None
        self.attr_matrix = None
        self.attr_matrix_norm = None

    def _get_path(self, fname):
        return os.path.join(self.path, self.name, fname)

    @property
    def n(self) -> int:
        if self._n:
            return self._n
        # 1. Use cache adj matrix
        if self.adj_matrix is not None:
            self._n = self.adj_matrix.shape[0]
        # 2: Use attribute file
        elif os.path.isfile(self._get_path('attribute.txt')):
            with open(self._get_path('attribute.txt'), 'r') as attr_f:
                nline = attr_f.readline().rstrip()
            self._n = int(''.join(filter(str.isdigit, nline)))
        # 3: Use label length | WARNING: incorrect for inductive dataset
        elif self.labels is not None:
            self._n = len(self.labels)
        return self._n

    @property
    def n_train(self) -> int:
        return len(self.idx_train)

    @property
    def n_val(self) -> int:
        return len(self.idx_val)

    @property
    def n_test(self) -> int:
        return len(self.idx_test)

    @property
    def m(self) -> int:
        if self._m:
            return self._m
        # 1: Use cache adj matrix
        if self.adj_matrix is not None:
            self._m = self.adj_matrix.nnz
        # 2: Use attribute file
        elif os.path.isfile(self._get_path('attribute.txt')):
            with open(self._get_path('attribute.txt'), 'r') as attr_f:
                nline = attr_f.readline().rstrip()
                mline = attr_f.readline().rstrip()
            self._m = int(''.join(filter(str.isdigit, mline)))
        # 3: Count by wc -l
        else:
            import subprocess
            self._m = int(subprocess.check_output(["wc", "-l", self.adjtxt_path]).split()[0])
        return self._m

    @property
    def nfeat(self) -> int:
        if self._nfeat:
            return self._nfeat
        if self.attr_matrix is None:
            self.input(['attr_matrix'])
        self._nfeat = self.attr_matrix.shape[1]
        return self._nfeat

    @property
    def nclass(self) -> int:
        if self._nclass:
            return self._nclass
        if self.labels is None:
            self.input(['labels'])
        # 1D array
        if self.labels.ndim == 1:
            # self._nclass = int(self.labels.max()) + 1
            lb_nonnan = self.labels[~np.isnan(self.labels)]
            self._nclass = int(lb_nonnan.max()) + 1
        # 2D one hot
        else:
            self._nclass = self.labels.shape[1]
        return self._nclass

    def __str__(self) -> str:
        s  = f"n={self.n}, m={self.m}, F={self.nfeat}, C={self.nclass} | "
        s += f"feat: {self.attr_matrix.shape}, label: {self.labels.shape} | "
        s += f"{self.n_train}/{self.n_val}/{self.n_test}="
        s += f"{self.n_train/self.n:0.2f}/{self.n_val/self.n:0.2f}/{self.n_test/self.n:0.2f}"
        return s

    def calculate(self, lst: List[str]) -> None:
        for key in lst:
            if key == 'deg':
                assert self.adj_matrix is not None
                assert np.sum(self.adj_matrix.diagonal()) == 0
                self.deg = self.adj_matrix.sum(1).A1
            elif key in ['idx_train', 'idx_val', 'idx_test']:
                # n_train = NTRAIN_PER_CLASS * self.nclass
                # n_val = NVAL_PER_CLASS * self.nclass
                n_train = int(0.50 * self.n)
                n_val = int(0.25 * self.n)
                if 'paper' == self.name:
                    np.random.seed(self.seed)
                    self.input(['idx_train', 'idx_val', 'idx_test'])
                    rnd = np.concatenate((self.idx_train, self.idx_val, self.idx_test))
                    rnd = np.random.permutation(rnd)
                    self.idx_train = np.sort(rnd[:n_train])
                    self.idx_val = np.sort(rnd[n_train:n_train + n_val])
                    self.idx_test = np.sort(rnd[n_train + n_val:])
                elif 'papers' in self.name:
                    # Common
                    n_train = NTRAIN_PER_CLASS * self.nclass
                    n_val = int(1.5 * NTRAIN_PER_CLASS * self.nclass)
                    self.input(['idx_train', 'idx_val', 'idx_test'])
                    rnd = np.concatenate((self.idx_train, self.idx_val, self.idx_test))
                    idxrange = n_train + n_val + 2000
                    rnd = np.random.permutation(rnd)[:idxrange]
                    idx_train, idx_val, idx_test = split_label(self.seed, len(rnd), NTRAIN_PER_CLASS, n_val, self.labels[rnd])
                    # NIGCN
                    # self.input(['idx_train', 'idx_val', 'idx_test'])
                    # rnd = np.concatenate((self.idx_train, self.idx_val, self.idx_test))
                    # n_train, n_val = 1000, 25000
                    # idxrange = n_train + n_val + 25000
                    # idx_train, idx_val, idx_test = split_random(self.seed, idxrange, n_train, n_val)

                    self.idx_train = np.sort(rnd[idx_train])
                    self.idx_val = np.sort(rnd[idx_val])
                    self.idx_test = np.sort(rnd[idx_test])
                elif 'mag' in self.name:
                    # self.idx_train, self.idx_val, self.idx_test = split_label(self.seed, self.n, NTRAIN_PER_CLASS * 5, n_val, self.labels)
                    self.idx_train, self.idx_val, self.idx_test = split_stratify(self.seed, self.n, n_train * 5, n_val, self.labels)
                elif 'ppi' in self.name or 'protein' in self.name or 'yelp' in self.name or 'amazon' in self.name:
                    self.idx_train, self.idx_val, self.idx_test = split_random(self.seed, self.n, n_train, n_val)
                else:
                    # self.idx_train, self.idx_val, self.idx_test = split_random(self.seed, self.n, n_train, n_val)
                    # self.idx_train, self.idx_val, self.idx_test = split_label(self.seed, self.n, NTRAIN_PER_CLASS, n_val, self.labels)
                    self.idx_train, self.idx_val, self.idx_test = split_stratify(self.seed, self.n, n_train, n_val, self.labels)
            elif key == 'labels_oh':
                if self.labels.ndim == 2:
                    self.labels_oh = self.labels
                else:
                    self.labels_oh = np.zeros((self.n, self.nclass), dtype=np.int8)
                    idx = ~ np.isnan(self.labels)
                    row = np.arange(self.labels.size)
                    self.labels_oh[row[idx], self.labels[idx]] = 1
            elif key == 'role':
                self.role = {}
                self.role['tr'] = self.idx_train.tolist()
                self.role['va'] = self.idx_val.tolist()
                self.role['te'] = self.idx_test.tolist()
            elif key == 'mask':
                self.mask = {}
                self.mask['tr'] = np.zeros(self.n, dtype=bool)
                self.mask['va'] = np.zeros(self.n, dtype=bool)
                self.mask['te'] = np.zeros(self.n, dtype=bool)
                self.mask['tr'][self.idx_train] = True
                self.mask['va'][self.idx_val] = True
                self.mask['te'][self.idx_test] = True
            elif key == 'edge_idx':
                self.edge_idx = self.adj_matrix.tocoo().copy()
                self.edge_idx = np.vstack([self.edge_idx.row, self.edge_idx.col])
            elif key == 'attr_matrix_norm':
                assert self.attr_matrix is not None
                assert self.deg is not None
                deg_pow = np.power(np.maximum(self.deg, 1e-12), 1 - self.rrz)
                deg_pow = diag_sp(deg_pow).astype(np.float32)
                self.attr_matrix_norm = deg_pow @ matstd(self.attr_matrix)
                self.attr_matrix_norm = matnorm_inf_dual(self.attr_matrix_norm).astype(np.float32)
                self.attr_matrix_norm = self.attr_matrix_norm.transpose().astype(np.float32, order='C')
            else:
                print("Key not exist: {}".format(key))

    def input(self, lst: List[str]) -> None:
        for key in lst:
            if key == 'adjnpz':
                self.adj_matrix = sp.load_npz(self.adjnpz_path)
                # assert self.adj_matrix.diagonal().sum() == 0, "adj_matrix error"
            elif key == 'adjtxt':
                with open(self.adjtxt_path, 'r') as attr_f:
                    nline = attr_f.readline().rstrip()
                self._n = int(''.join(filter(str.isdigit, nline)))
                adjtxt = np.loadtxt(self.adjtxt_path)
                self._m = adjtxt.shape[0]
                ones = np.ones((self.m), dtype=np.int8)
                self.adj_matrix = sp.coo_matrix(
                    (ones, (adjtxt[:, 0], adjtxt[:, 1])),
                    shape=(self.n, self.n))
                self.adj_matrix = self.adj_matrix.tocsr()
                # assert self.adj_matrix.diagonal().sum() == 0, "adj_matrix error"
            elif key == 'deg':
                #self.deg = dict(np.load(self.degree_path))['arr_0']
              
                data = np.load(self.degree_path, allow_pickle=True)
                self.deg = data[next(iter(data.keys()))]
            elif key == 'labels':
                self.labels = dict(np.load(self.labels_path, allow_pickle=True))['labels']
                # assert (self.labels.dim()==2 and self.labels.shape[1]==1) or self.labels.dim()==1, "label shape error"
            elif key == 'idx_train':
                self.idx_train = dict(np.load(self.labels_path, allow_pickle=True))['idx_train']
            elif key == 'idx_val':
                self.idx_val = dict(np.load(self.labels_path, allow_pickle=True))['idx_val']
            elif key == 'idx_test':
                self.idx_test = dict(np.load(self.labels_path, allow_pickle=True))['idx_test']
            elif key == 'attr_matrix':
                self.attr_matrix = np.load(self.feats_path)
            elif key == 'attr_matrix_norm':
                self.attr_matrix_norm = np.load(self.featsnorm_path)
            else:
                print("Key not exist: {}".format(key))

    def output(self, lst: List[str]) -> None:
        for key in lst:
            if key == 'adjnpz':
                self.adj_matrix = self.adj_matrix.tocsr()
                assert sp.isspmatrix_csr(self.adj_matrix)
                sp.save_npz(self.adjnpz_path, self.adj_matrix)
            elif key == 'adjtxt':
                self.adj_matrix = self.adj_matrix.tocoo()
                with open(self.adjtxt_path, 'w') as f:
                    f.write("# {:d}\n".format(self.n))
                    for i in range(self.m):
                        f.write("{:d} {:d}\n".format(self.adj_matrix.row[i], self.adj_matrix.col[i]))
            elif key == 'adjl':
                import struct
                self.adj_matrix = self.adj_matrix.tocsr()
                assert sp.isspmatrix_csr(self.adj_matrix), "Adj wrong format"
                el = self.adj_matrix.indices
                pl = self.adj_matrix.indptr
                el = np.array(el, dtype=np.uint32)
                pl = np.array(pl, dtype=np.uint32)

                el_re = []
                for i in range(1,pl.shape[0]):
                    el_re += sorted(el[pl[i-1]:pl[i]],key=lambda x:pl[x+1]-pl[x])
                el = np.asarray(el_re,dtype=np.uint32)

                with open(self._get_path('adj_el.bin'), 'wb') as f:
                    for i in el:
                        m = struct.pack('I', i)
                        f.write(m)
                with open(self._get_path('adj_pl.bin'), 'wb') as f:
                    for i in pl:
                        m = struct.pack('I', i)
                        f.write(m)
            elif key == 'attribute':
                with open(self._get_path("attribute.txt"), 'w') as f:
                    f.write(f"n={self.n:d}\n")
                    f.write(f"m={self.m:d}")
            elif key == 'deg':
                np.savez_compressed(self.degree_path, self.deg)
            elif key in ['labels', 'idx_train', 'idx_val', 'idx_test']:
                labels_dict = {'labels': self.labels,
                               'idx_train': self.idx_train,
                               'idx_val': self.idx_val,
                               'idx_test': self.idx_test}
                np.savez_compressed(self.labels_path, **labels_dict)
            elif key == 'query':
                query = np.arange(self.n, dtype=int)
                np.savetxt(self.query_path, query, fmt='%d', delimiter='\n')
            elif key == 'query_topdeg':
                k = self.nfeat
                idx = np.argpartition(self.deg, -k)[-k:]
                np.savetxt(self.query_path, idx, fmt='%d', delimiter='\n')
            elif key == 'query_train':
                assert self.idx_train is not None
                np.savetxt(self.querytrain_path, self.idx_train, fmt='%d', delimiter='\n')
            elif key == 'attr_matrix':
                self.attr_matrix = self.attr_matrix.astype(np.float32, order='C')
                np.save(self.feats_path, self.attr_matrix)
            elif key == 'attr_matrix_norm':
                self.attr_matrix_norm = self.attr_matrix_norm.astype(np.float32, order='C')
                np.save(self.featsnorm_path, self.attr_matrix_norm)
            else:
                print("Key not exist: {}".format(key))

    def output_split(self, attr_matrix: Optional[np.ndarray], spt: int=10, name: str='feats') -> None:
        """Split large matrix by feature dimension."""
        from tqdm import trange
        if attr_matrix is None:
            attr_matrix = self.attr_matrix
        n = attr_matrix.shape[0]
        nd = n // spt
        for i in trange(spt):
            if i < spt - 1:
                idxl, idxr = i * nd, (i+1) * nd
            else:
                idxl, idxr = i * nd, n
            prt = attr_matrix[idxl:idxr, :]

            prt_path = self._get_path('{}_{}.npy'.format(name, i))
            np.save(prt_path, prt)

    def to_undirected(self) -> None:
        self.adj_matrix = self.adj_matrix + self.adj_matrix.T
        self.adj_matrix.data = np.ones(len(self.adj_matrix.data), dtype=np.int8)
        self._m = self.adj_matrix.nnz

        self.calculate(['deg'])
        idx_zero = np.where(self.deg == 0)[0]
        if len(idx_zero) > 0:
            print(f"Warning: {len(idx_zero)} isolated nodes found: {idx_zero}!")


class DataProcess_inductive(DataProcess):
    """ Inductive processor for generating adj and attr of only train nodes
    For graphs with edge attributes refer to pyg.utils.subgraph
    """
    def  __init__(self, name: str, path: str = '../data/', rrz: float = 0.5, seed: int = 0) -> None:
        name = name + '_train'
        super().__init__(name, path, rrz, seed)

    def fetch(self) -> None:
        # Node index mapping
        assert self.adj_matrix is not None
        assert self.idx_train is not None
        nraw = self.adj_matrix.shape[0]
        ntrain = self.n_train
        mapping = - np.ones(nraw, dtype=int)   # key: old index, value: 0~ntrain-1
        mapping[self.idx_train] = np.arange(ntrain)

        # Construct induced adj
        self.adj_matrix = self.adj_matrix.tocoo()
        idx_in = np.isin(self.adj_matrix.col, self.idx_train) & np.isin(self.adj_matrix.row, self.idx_train)
        col, row = mapping[self.adj_matrix.col[idx_in]], mapping[self.adj_matrix.row[idx_in]]
        ones = np.ones(len(col), dtype=np.int8)
        self.adj_matrix = sp.coo_matrix(
            (ones, (col, row)),
            shape=(ntrain, ntrain))

        self.adj_matrix = self.adj_matrix.tocsr()
        self.adj_matrix.setdiag(0)
        self.adj_matrix.eliminate_zeros()
        self.adj_matrix.data = np.ones(len(self.adj_matrix.data), dtype=np.int8)
        self._n, self._m = ntrain, self.adj_matrix.nnz

        # Construct induced attr
        assert self.attr_matrix is not None
        self.attr_matrix = self.attr_matrix[self.idx_train]
        self.attr_matrix = self.attr_matrix.astype(np.float32, order='C')


if __name__ == '__main__':
    dp = DataProcess('pubmed', seed=0)
    dp.input(['adjtxt', 'attr_matrix', 'labels'])
    dp.calculate(['deg', 'idx_train'])
    dp.output(['deg', 'query'])
    print(dp)

