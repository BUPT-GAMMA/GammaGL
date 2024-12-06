import pandas as pd
import numpy as np
import scipy.sparse as sp
from scipy.spatial import distance_matrix
import os.path as osp
from typing import Union, List, Tuple
from gammagl.data import download_url, InMemoryDataset, Graph
import tensorlayerx as tlx
from gammagl.utils.mask import index_to_mask


def sys_normalized_adjacency(adj):
    adj = sp.coo_matrix(adj)
    adj = adj + sp.eye(adj.shape[0])
    row_sum = np.array(adj.sum(1))
    row_sum = (row_sum == 0) * 1 + row_sum
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


def build_relationship(x, thresh=0.25):
    df_euclid = pd.DataFrame(
        1 / (1 + distance_matrix(x.T.T, x.T.T)), columns=x.T.columns, index=x.T.columns)
    df_euclid = df_euclid.to_numpy()
    idx_map = []
    for ind in range(df_euclid.shape[0]):
        max_sim = np.sort(df_euclid[ind, :])[-2]
        neig_id = np.where(df_euclid[ind, :] > thresh * max_sim)[0]
        import random
        random.seed(912)
        random.shuffle(neig_id)
        for neig in neig_id:
            if neig != ind:
                idx_map.append([ind, neig])
    idx_map = np.array(idx_map)
    return idx_map


class Credit(InMemoryDataset):
    r"""
    The datasets "Bail-Bs" from the
    `"Graph Fairness Learning under Distribution Shifts"
    <https://arxiv.org/abs/2401.16784>`_ paper.
    Nodes represent credit card users.
    Training, validation and test splits are given by binary masks.
    
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
    force_reload (bool, optional): Whether to re-process the dataset.
        (default: :obj:`False`)

    Tip
    ---
        .. list-table::
            :widths: 10 10 10 10 10
            :header-rows: 1
            
            * - Name
              - #nodes
              - #edges
              - #classes
            * - Credit_C0
              - 4184
              - 45718
              - 13
              - 2
            * - Credit_C1
              - 2541
              - 18949
              - 13
              - 2
            * - Credit_C2
              - 3796
              - 28936
              - 13
              - 2
            * - Credit_C3
              - 2068
              - 15314
              - 13
              - 2
            * - Credit_C4
              - 3420
              - 26048
              - 13
              - 2
    """

    url = 'https://raw.githubusercontent.com/liushiliushi/FatraGNN/main/dataset'


    def __init__(self, root=None, transform=None, pre_transform=None, pre_filter=None, force_reload: bool = False):
        self.name = 'credit'
        self.top_k = 10
        self.strlist = ['_C0', '_C1', '_C2', '_C3', '_C4']
        super(Credit, self).__init__(root, transform, pre_transform, pre_filter, force_reload = force_reload)
        self.data = self.load_data(self.processed_paths[0])
        
    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        self.strlist = ['_C0', '_C1', '_C2', '_C3', '_C4']
        feature_names = [f'{self.name}{name}.csv' for name in self.strlist]
        edge_names = [f'{self.name}{name}_edges.txt' for name in self.strlist]
        return feature_names + edge_names

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return tlx.BACKEND + '_data.pt'

    @property
    def num_classes(self) -> int:
        return super().num_classes()

    def download(self):
        for name in self.raw_file_names:
            download_url(f'{self.url}/{self.name}/{name}', self.raw_dir)

    def process(self):
        sens_attr="Age"
        predict_attr="NoDefaultNextMonth"
        label_number=6000

        data_list = []
        for i in self.strlist:

            idx_features_labels = pd.read_csv(osp.join(self.raw_dir, "{}.csv".format(self.name + i)))
            if 'Unnamed: 0' in idx_features_labels.columns:
                idx_features_labels = idx_features_labels.drop(['Unnamed: 0'], axis=1)

            header = list(idx_features_labels.columns)
            header.remove(predict_attr)
            header.remove('Single')

            edges_unordered = np.genfromtxt(osp.join(self.raw_dir, "{}_edges.txt".format(self.name + i))).astype('int')
            features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
            labels = idx_features_labels[predict_attr].values

            idx = np.arange(features.shape[0])
            idx_map = {j: i for i, j in enumerate(idx)}
            edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                            dtype=int).reshape(edges_unordered.shape)
            adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                                shape=(labels.shape[0], labels.shape[0]),
                                dtype=np.float32)

            adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
            adj = adj + sp.eye(adj.shape[0])
            adj_norm = sys_normalized_adjacency(adj)

            edge_index = tlx.convert_to_tensor(np.vstack((adj_norm.row, adj_norm.col)).astype(np.int64)) 
            features = tlx.convert_to_tensor(np.array(features.todense()).astype(np.float32))
            
            import random
            random.seed(20)
            label_idx_0 = np.where(labels == 0)[0]
            label_idx_1 = np.where(labels == 1)[0]

            labels = tlx.convert_to_tensor(labels.astype(np.float32))
            random.shuffle(label_idx_0)
            random.shuffle(label_idx_1)
            idx_train = np.append(label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number // 2)],
                                label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number // 2)])
            idx_val = np.append(label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(
                label_idx_0))], label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))])
            idx_test = np.append(label_idx_0[int(
                0.75 * len(label_idx_0)):], label_idx_1[int(0.75 * len(label_idx_1)):])

            sens = idx_features_labels[sens_attr].values.astype(int)
            sens = tlx.convert_to_tensor(sens)
            train_mask = index_to_mask(tlx.convert_to_tensor(idx_train), features.shape[0])
            val_mask = index_to_mask(tlx.convert_to_tensor(idx_val), features.shape[0])
            test_mask = index_to_mask(tlx.convert_to_tensor(idx_test), features.shape[0])

            sens_idx = 1
            x_max, x_min = tlx.reduce_max(features, axis=0), tlx.reduce_min(features, axis=0)
            
            norm_features = 2 * (features - x_min)/(x_max - x_min) - 1
            norm_features = tlx.convert_to_numpy(norm_features)
            features = tlx.convert_to_numpy(features)
            norm_features[:, sens_idx] = features[:, sens_idx]
            features = norm_features
            features = tlx.convert_to_tensor(features)
            corr = pd.DataFrame(np.array(tlx.to_device(features, 'cpu'))).corr()
            corr_matrix = corr[sens_idx].to_numpy()
            corr_idx = np.argsort(-np.abs(corr_matrix))
            corr_idx = tlx.convert_to_tensor(corr_idx[:self.top_k])

            data = Graph(x=features, edge_index=edge_index, adj=adj, y=labels,
                          train_mask=train_mask, val_mask=val_mask, test_mask=test_mask, sens=sens)
            data_list.append(data)
        self.save_data(data_list, self.processed_paths[0])

