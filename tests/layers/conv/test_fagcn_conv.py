# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   test_fagcn_conv.py
@Time    :   2022/5/20 10:44
@Author  :   Ma Zeyao
"""

import os
os.environ['TL_BACKEND'] = 'torch'
# os.environ['TL_BACKEND'] = 'tensorflow'
# os.environ['TL_BACKEND'] = 'paddle'
# os.environ['TL_BACKEND'] = 'mindspore'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorlayerx as tlx
from gammagl.layers.conv import MessagePassing

def test_fagcn():
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
