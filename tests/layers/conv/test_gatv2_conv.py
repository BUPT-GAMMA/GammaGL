# !/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2022/06/17 09:05
# @Author  : clear
# @FileName: test_gatv2-conv.py

import tensorlayerx as tlx
from gammagl.layers.conv import GATV2Conv


def test_gatv2_conv():
    x = tlx.random_normal(shape=(4, 64))
    edge_index = tlx.convert_to_tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])

    conv = GATV2Conv(in_channels=64, out_channels=16)
    out = conv(x, edge_index)
    assert tlx.get_tensor_shape(out) == [4, 16]
