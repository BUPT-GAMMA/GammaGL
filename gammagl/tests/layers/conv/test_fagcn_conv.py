# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   test_fagcn_conv.py
@Time    :   2022/5/20 10:44
@Author  :   Ma Zeyao
"""

import tensorlayerx as tlx
import numpy as np
from gammagl.layers.conv import FAGCNConv
from gammagl.utils import calc_gcn_norm

def test_fagcn():
    x = np.random.uniform(low = 0, high = 1, size = (4, 16))
    x = tlx.convert_to_tensor(x, dtype = tlx.float32)
    edge_index = tlx.convert_to_tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    edge_weight = tlx.convert_to_tensor(calc_gcn_norm(edge_index, x.shape[0]))

    conv = FAGCNConv(hidden_dim=16, drop_rate=0.6)
    out = conv(x, edge_index, edge_weight, x.shape[0])

    assert tlx.get_tensor_shape(out) == [4, 16]
