import os.path as osp
from time import time
import torch
from torch_geometric.datasets import Reddit

import quiver

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Reddit')
dataset = Reddit(path)
data = dataset[0]
train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)
################################
# Step 1: Using Quiver's sampler
################################

train_loader = torch.utils.data.DataLoader(train_idx,
                                           batch_size=1024,
                                           shuffle=True,
                                           drop_last=True) # Quiver
csr_topo = quiver.CSRTopo(data.edge_index) # Quiver
quiver_sampler = quiver.pyg.GraphSageSampler(csr_topo, sizes=[25, 10], device=0, mode='GPU') # Quiver

################################
# Step 2: Using Quiver's Feature
################################
# x = data.x.to(device) # Original PyG Code
x = quiver.Feature(rank=0, device_list=[0], device_cache_size=".1G", cache_policy="device_replicate", csr_topo=csr_topo) # Quiver
x.from_cpu_tensor(data.x) # Quiver




def train(epoch):
    st = time()
    for seeds in train_loader: # Quiver
        n_id, batch_size, adjs = quiver_sampler.sample(seeds) # Quiver
    ed = time()
for epoch in range(1, 11):
    costs = train(epoch)
    print(f'Epoch {epoch:02d}, Time: {costs:.4f}')
