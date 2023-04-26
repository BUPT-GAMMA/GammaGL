# !/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2022/06/17 09:14
# @Author  : clear
# @FileName: test_gcnii_conv.py

import tensorlayerx as tlx
import numpy as np
from gammagl.layers.conv import GCNIIConv
from gammagl.utils import calc_gcn_norm

def test_gcnii():
    x = np.random.uniform(low = 0, high = 1, size = (4, 16))
    x = tlx.convert_to_tensor(x, dtype = tlx.float32)
    edge_index = tlx.convert_to_tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    edge_weight = tlx.convert_to_tensor(calc_gcn_norm(edge_index, x.shape[0]))

    conv1 = GCNIIConv(in_channels=16, out_channels=16, alpha=0.1, beta=0.5, variant=False)
    conv2 = GCNIIConv(in_channels=16, out_channels=16, alpha=0.1, beta=0.5, variant=True)
    x0 = x
    x = conv1(x0, x, edge_index, edge_weight=edge_weight, num_nodes=4)
    out = conv2(x0, x, edge_index, edge_weight=edge_weight, num_nodes=4)

    assert tlx.get_tensor_shape(out) == [4, 16]
