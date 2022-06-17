# !/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2022/05/12 21:14
# @Author  : clear
# @FileName: test_gat_conv.py

import tensorlayerx as tlx
from gammagl.layers.conv import GATConv


def test_gat_conv():
    x = tlx.random_normal(shape=(4, 64))
    edge_index = tlx.convert_to_tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])

    conv = GATConv(in_channels=64, out_channels=16)
    out = conv(x, edge_index)
    assert tlx.get_tensor_shape(out) == [4, 16]
