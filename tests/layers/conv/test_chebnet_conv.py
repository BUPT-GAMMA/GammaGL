# !/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2022/06/17 09:14
# @Author  : clear
# @FileName: test_chebnet_conv.py


import tensorlayerx as tlx
from gammagl.layers.conv import ChebConv
from gammagl.utils import calc_gcn_norm

def test_chebnet():
    x = tlx.random_uniform(shape=(4, 16))
    edge_index = tlx.convert_to_tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    edge_weight = tlx.convert_to_tensor(calc_gcn_norm(edge_index, x.shape[0]))

    conv = ChebConv(in_channels=16, out_channels=8, K=2)
    out = conv(x, edge_index, num_nodes=x.shape[0], edge_weight=edge_weight)

    assert out.shape == (4, 8)

# test_chebnet()
