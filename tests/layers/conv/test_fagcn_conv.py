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
import numpy as np
import tensorlayerx as tlx
from gammagl.layers.conv import FAGCNConv


def test_fagcn():
    x = tlx.random_uniform(shape=(4, 16))
    edge_index = tlx.convert_to_tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    src_degree = tlx.convert_to_tensor(np.power([3, 3, 3, 1, 1, 1], -0.5), dtype='float32')
    dst_degree = tlx.convert_to_tensor(np.power([1, 1, 1, 3, 3, 3], -0.5), dtype='float32')

    conv = FAGCNConv(src_degree=src_degree, dst_degree=dst_degree, hidden_dim=16, drop_rate=0.6)
    out = conv(x, edge_index, x.shape[0])

    assert out.shape == (4, 16)
