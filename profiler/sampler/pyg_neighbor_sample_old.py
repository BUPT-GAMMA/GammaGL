import os.path as osp
from time import time
import numpy as np
from torch_geometric.datasets import Reddit
from torch_geometric.loader import NeighborSampler


path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'reddit')
dataset = Reddit(path)
data = dataset[0]

train_loader = NeighborSampler(data.edge_index, node_idx=data.train_mask,
                               sizes=[25, 10], batch_size=1024, shuffle=True,
                               num_workers=0)
subgraph_loader = NeighborSampler(data.edge_index, node_idx=None, sizes=[-1],
                                  batch_size=1024, shuffle=False,
                                  num_workers=0)
"""
test sample without replacement speed
"""
arrs = []
for i in range(10):
    a = time()
    for batch_size, n_id, adjs in train_loader:
        pass
    b = time()
    if i > 3:
        arrs.append(b - a)
print("rand sample costs:{} \u00B1 {} s".format(np.mean(arrs), np.std(arrs)))
"""
test sample without replacement speed
"""
arrs = []
for i in range(10):
    a = time()
    for batch_size, n_id, adjs in subgraph_loader:
        pass
    b = time()
    if i > 3:
        arrs.append(b - a)
print("rand sample costs:{} \u00B1 {} s".format(np.mean(arrs), np.std(arrs)))