import os
# os.environ['TL_BACKEND'] = 'paddle'
# os.environ['CUDA_VISIBLE_DEVICES'] = ' '
# set your backend here, default `tensorflow`
from time import time

from gammagl.utils import mask_to_index
import numpy as np
from gammagl.datasets import Reddit
import tensorlayerx as tlx
import argparse
from gammagl.loader.neighbor_sampler import NeighborSampler
# load reddit dataset
dataset = Reddit()
# dataset.process()  # suggest to execute explicitly so far
graph = dataset[0]

train_idx = mask_to_index(graph.train_mask)

train_loader = NeighborSampler(edge_index=graph.edge_index,
                               node_idx=graph.train_mask
                               , sample_lists=[25, 10]
                               , batch_size=1024
                               , shuffle=False
                               , num_workers=0)

sub_loader = NeighborSampler(edge_index=graph.edge_index,
                             node_idx=None,
                             sample_lists=[-1],
                             batch_size=1024
                             , shuffle=False
                             , num_workers=0)

"""
test sample without replacement speed
"""
arrs = []
for i in range(10):
    aa = time()
    for batch, n_id, adjs in train_loader:
        pass
    bb = time()
    if i > 3:
        arrs.append(bb - aa)
print("rand sample costs:{} \u00B1 {} s".format(np.mean(arrs), np.std(arrs)))

"""
test sample without replacement speed
"""
arrs = []
for i in range(10):
    aa = time()
    for dst_node, adjs, all_node in sub_loader:
        pass
    bb = time()
    if i > 3:
        arrs.append(bb - aa)
print("full sample costs:{} \u00B1 {} s".format(np.mean(arrs), np.std(arrs)))

# rand sample costs:19.910096804300945 +- 0.25383039604097435 s
# full sample costs:15.682573000590006 +- 1.6219983738035213 s
