# !/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2022/06/17 09:05
# @Author  : clear
# @FileName: test_appnp_conv.py.py

import tensorlayerx as tlx
from gammagl.layers.conv import APPNPConv
from gammagl.utils import calc_gcn_norm

def test_appnp():
    x = tlx.random_uniform(shape=(4, 16))
    edge_index = tlx.convert_to_tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    edge_weight = tlx.convert_to_tensor(calc_gcn_norm(edge_index, x.shape[0]))

    conv = APPNPConv(in_channels=16, out_channels=8, iter_K=5, alpha=0.1, drop_rate=0.6)
    out = conv(x, edge_index=edge_index, edge_weight=edge_weight)

    assert out.shape == (4, 8)

# test_appnp()
