# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   test_fagcn_conv.py
@Time    :   2022/5/20 10:44
@Author  :   Ma Zeyao
"""

import tensorlayerx as tlx
from gammagl.layers.conv import FAGCNConv
from gammagl.utils import calc_gcn_norm

def test_fagcn():
    x = tlx.random_uniform(shape=(4, 16))
    edge_index = tlx.convert_to_tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    edge_weight = tlx.convert_to_tensor(calc_gcn_norm(edge_index, x.shape[0]))

    conv = FAGCNConv(hidden_dim=16, drop_rate=0.6)
    out = conv(x, edge_index, edge_weight, x.shape[0])

    assert out.shape == (4, 16)
