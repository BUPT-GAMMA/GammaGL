# !/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2022/06/17 09:14
# @Author  : clear
# @FileName: test_chebnet_conv.py


import tensorlayerx as tlx
import numpy as np
from gammagl.layers.conv import ChebConv
from gammagl.utils import calc_gcn_norm


def test_chebnet():
    x = np.random.uniform(low = 0, high = 1, size = (4, 16))
    x = tlx.convert_to_tensor(x, dtype = tlx.float32)
    edge_index = tlx.convert_to_tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    edge_weight = tlx.convert_to_tensor(calc_gcn_norm(edge_index, x.shape[0]))

    conv = ChebConv(in_channels=16, out_channels=8, K=2)
    out = conv(x, edge_index, num_nodes=x.shape[0], edge_weight=edge_weight)

    assert tlx.get_tensor_shape(out) == [4, 8]
