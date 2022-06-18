# !/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2022/06/17 09:15
# @Author  : clear
# @FileName: test_gprgnn_conv.py

import tensorlayerx as tlx
from gammagl.layers.conv import GPRConv


def test_gprgnn_conv():
    x = tlx.random_normal(shape=(4, 64))
    edge_index = tlx.convert_to_tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])

    conv = GPRConv(K=2, alpha=0.5)
    out = conv(x, edge_index)
    assert tlx.get_tensor_shape(out) == [4, 64]
