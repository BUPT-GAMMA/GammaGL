# !/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2022/06/17 09:05
# @Author  : clear
# @FileName: test_sgc_conv.py.py

import tensorlayerx as tlx
from gammagl.layers.conv import SGConv

def test_sgc_conv():
    x = tlx.random_normal(shape=(4, 64))
    edge_index = tlx.convert_to_tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])

    conv = SGConv(iter_K=3, in_channels=64, out_channels=16)
    out = conv(x, edge_index)
    assert tlx.get_tensor_shape(out) == [4, 16]

