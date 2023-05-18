# -*- coding: utf-8 -*-
# @author WuJing
# @created 2023/4/8

import os
# os.environ['TL_BACKEND'] = 'paddle'
# os.environ['CUDA_VISIBLE_DEVICES'] = ' '
# set your backend here, default `tensorflow`
from time import time

import copy

from gammagl.utils import mask_to_index
import numpy as np
from gammagl.datasets import Reddit
import tensorlayerx as tlx
import argparse
# from gammagl.loader.Neighbour_sampler import Neighbor_Sampler
from gammagl.loader.node_neighbor_loader import NodeNeighborLoader
from gammagl.utils.unit_test_utils import Timer

# load reddit dataset
dataset = Reddit()
# dataset.process()  # suggest to execute explicitly so far
graph = dataset[0]

train_idx = mask_to_index(graph.train_mask)

train_loader = NodeNeighborLoader(graph=graph
                                  , input_nodes_type=graph.train_mask
                                  , num_neighbors=[25, 10]
                                  , batch_size=1024
                                  , shuffle=False
                                  )

sub_loader = NodeNeighborLoader(copy.copy(graph), input_nodes_type=None,
                                num_neighbors=[-1]
                                , batch_size=1024
                                , shuffle=False
                                )


"""
test sample without replacement speed
"""
arrs = []
for i in range(10):
    aa = time()
    for batch in train_loader:
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
    for batch in sub_loader:
        pass
    bb = time()
    if i > 3:
        arrs.append(bb - aa)
print("full sample costs:{} \u00B1 {} s".format(np.mean(arrs), np.std(arrs)))
