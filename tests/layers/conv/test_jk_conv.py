# !/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2022/06/20 13:23
# @Author  : clear
# @FileName: test_jk_conv.py

import tensorlayerx as tlx
from gammagl.layers.conv import JumpingKnowledge


def test_jk_conv():
    xs = [tlx.random_normal(shape=(4, 16)) for _ in range(4)]
    # edge_index = tlx.convert_to_tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])

    conv_cat = JumpingKnowledge(mode='cat', channels=16, num_layers=4)
    out_cat = conv_cat(xs)
    assert tlx.get_tensor_shape(out_cat) == [4, 64]

    conv_lstm = JumpingKnowledge(mode='lstm', channels=16, num_layers=4)
    out_lstm = conv_lstm(xs)
    assert tlx.get_tensor_shape(out_lstm) == [4, 16]

    conv_max = JumpingKnowledge(mode='max', channels=16, num_layers=4)
    out_max = conv_max(xs)
    assert tlx.get_tensor_shape(out_max) == [4, 16]

