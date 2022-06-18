import copy
import os.path as osp
from time import time
import numpy as np
import torch
from torch_geometric.datasets import Reddit
from torch_geometric.loader import NeighborLoader


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'reddit')
dataset = Reddit(path)

# Already send node features/labels to GPU for faster access during sampling:
data = dataset[0].to(device, 'x', 'y')

kwargs = {'batch_size': 1024, 'num_workers': 0}
train_loader = NeighborLoader(data, input_nodes=data.train_mask,
                              num_neighbors=[25, 10], shuffle=True, **kwargs)

subgraph_loader = NeighborLoader(copy.copy(data), input_nodes=None,
                                 num_neighbors=[-1], shuffle=False, **kwargs)



"""
test sample without replacement speed
"""
arrs = []
for i in range(10):

    a = time()
    for data in train_loader:
        pass
    b = time()
    if i > 3:
        arrs.append(b - a)
print("rand sample costs:{} \u00B1 {} s".format(np.mean(arrs), np.std(arrs)))
# rand sample costs:25.175019303957622 +- 1.6890655608040088 s
"""
test sample without replacement speed
"""
arrs = []
for i in range(10):

    a = time()
    for data in subgraph_loader:
        pass
    b = time()
    if i > 3:
        arrs.append(b - a)
print("full sample costs:{} \u00B1 {} s".format(np.mean(arrs), np.std(arrs)))

# full sample costs:30.793972452481587 +- 0.717800055861003 s

