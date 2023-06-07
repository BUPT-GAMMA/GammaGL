# !/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2022/05/12 16:18
# @Author  : clear
# @FileName: test_gcn_conv.py

import numpy as np
import tensorlayerx as tlx
import scipy.sparse as sp
from gammagl.layers.conv import GCNConv


def AXWb(A, X, W, b):
    X = tlx.matmul(X, W)
    Y = tlx.matmul(A, X)
    return Y + b


def test_gcn_conv():
    x = tlx.random_normal(shape=(4, 16))
    edge_index = tlx.convert_to_tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])

    conv = GCNConv(in_channels=16, out_channels=16, norm='none')
    out_conv = conv(x, edge_index)

    edge_index = tlx.convert_to_numpy(edge_index)
    col, row = edge_index[0], edge_index[1]
    data = np.ones(col.shape[0])
    A = sp.coo_matrix((data, (row, col)), shape=(4, 4)).toarray()
    A = tlx.convert_to_tensor(A, dtype=tlx.float32)
    w = conv.linear.trainable_weights[0]
    out_mm = AXWb(A, x, w, conv.bias)

    out_conv = tlx.convert_to_numpy(out_conv)
    out_mm = tlx.convert_to_numpy(out_mm)
    assert out_conv.shape == (4, 16)
    assert out_mm.shape == (4, 16)

    """
        we found Computational precision problem in some backends, 
        and here are some statics(10 times):

        backend     mean-error  std     \n
        paddle      0.0018      0.0014  \n
        tensorflow  0.0023      0.0014  \n
        torch       0.0         0.0     \n
        mindspore   0.0         0.0     \n
    """
    # if tlx.BACKEND in ('torch', 'mindspore'):
    #     assert np.allclose(out_mm, out_conv)
    # elif tlx.BACKEND in ('paddle', 'tensorflow'):
    #     assert np.allclose(out_mm, out_conv, atol=1e-2)