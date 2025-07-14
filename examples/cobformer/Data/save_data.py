import numpy as np
import os.path as osp

import scipy.sparse

from get_data import get_data
import torch
from torch_sparse import SparseTensor
from torch_geometric.utils import degree


dataset = 'ogbn-products'
path = osp.join(osp.expanduser('~'), 'datasets', dataset)
data = get_data(path, dataset)
N = data.graph['num_nodes']
edge_index = data.graph['edge_index']
row, col = edge_index
d = degree(col, N).float()
d_norm_in = (1. / d[col]).sqrt()
d_norm_out = (1. / d[row]).sqrt()
value = torch.ones_like(row) * d_norm_in * d_norm_out
value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
adj = scipy.sparse.coo_matrix((value, (row.numpy(), col.numpy())), shape=[N,N])
feature = data.graph['node_feat']
labels = data.label
split_list = []
if dataset in ['film', 'deezer']:
    for i in range(10):
        splits_file_path = '{}/{}'.format(path, dataset) + '_split_50_25_' + str(i) + '.npz'
        with np.load(splits_file_path) as splits_file:
            idx_train = torch.BoolTensor(splits_file['train_mask']).nonzero().squeeze()
            idx_val = torch.BoolTensor(splits_file['val_mask']).nonzero().squeeze()
            idx_test = torch.BoolTensor(splits_file['test_mask']).nonzero().squeeze()
        split_list.append([idx_train, idx_val, idx_test])
else:
    idx_train = data.train_mask.nonzero().squeeze()
    idx_val = data.valid_mask.nonzero().squeeze()
    idx_test = data.test_mask.nonzero().squeeze()
    split_list.append([idx_train, idx_val, idx_test])
data_list = [adj, feature, labels] + split_list
torch.save(data_list, f'data4NAG/{dataset}.pt')
print("save done")
