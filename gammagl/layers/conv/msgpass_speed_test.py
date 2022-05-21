# !/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2022/04/21 06:23
# @Author  : clear
# @FileName: msgpass_speed_test.py.py

import os
os.environ['TL_BACKEND'] = 'paddle' # set your backend here, default `tensorflow`
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
sys.path.insert(0, os.path.abspath('../../')) # adds path2gammagl to execute in command line.
import argparse
import tensorlayerx as tlx
from gammagl.datasets import Planetoid
from gammagl.models import GCNModel
from gammagl.utils.loop import add_self_loops
from tensorlayerx.model import TrainOneStep, WithLoss

dataset = Planetoid(r'../', 'cora')
dataset.process()
graph = dataset[0]
graph.tensor()
edge_index, _ = add_self_loops(graph.edge_index, n_loops=1)
edge_weight = tlx.ops.convert_to_tensor(GCNModel.calc_gcn_norm(edge_index, graph.num_nodes))
x = graph.x
y = tlx.argmax(graph.y, axis=1)


