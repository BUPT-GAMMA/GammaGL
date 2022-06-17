# !/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2022/05/22 15:50
# @Author  : clear
# @FileName: test_message_passing.py

import tensorlayerx as tlx
from gammagl.layers.conv import MessagePassing


def test_message_passing():
    x = tlx.reshape(tlx.arange(0, 8, dtype=tlx.float32), shape=(4,2))
    edge_index = tlx.convert_to_tensor([[0, 1, 2, 2, 3],
                                        [1, 2, 1, 0, 3]])

    mp = MessagePassing()
    sum_x = mp.propagate(x, edge_index, aggr='sum')
    mean_x = mp.propagate(x, edge_index, aggr='mean')
    max_x = mp.propagate(x, edge_index, aggr='max')

    assert tlx.ops.convert_to_numpy(sum_x).tolist() == [[4.0, 5.0], [4.0, 6.0], [2.0, 3.0], [6.0, 7.0]]
    assert tlx.ops.convert_to_numpy(mean_x).tolist() == [[4.0, 5.0], [2.0, 3.0], [2.0, 3.0], [6.0, 7.0]]
    assert tlx.ops.convert_to_numpy(max_x).tolist() == [[4.0, 5.0], [4.0, 5.0], [2.0, 3.0], [6.0, 7.0]]

# test_message_passing()
