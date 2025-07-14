import torch

from Data.get_data import *
import os.path as osp

dataset = 'ogbn-products'
# dataset = 'Cora'
# n_patch = 16384
n_patch = 8192
# n_patch = 112
path = osp.join(osp.expanduser('~'), 'datasets', dataset)
data = get_data(path, dataset)

patch = metis_partition(data.graph, n_patch)
# node_mask = torch.load('partition/'+dataset+'_partition_{}.pt')
# patch = patch2batch(data.graph, node_mask)
torch.save(patch, 'Data/partition/'+dataset+f'_partition_{n_patch}.pt')
print('Done!!!')
