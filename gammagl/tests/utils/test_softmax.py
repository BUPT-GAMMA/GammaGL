# !/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2022/05/23 15:27
# @Author  : clear
# @FileName: test_softmax.py

import tensorlayerx as tlx
import numpy as np
from gammagl.utils.softmax import segment_softmax


def test_softmax():
    x = tlx.convert_to_tensor([[1,1], [1,1], [2,4], [2,4]], dtype=tlx.float32)
    edge_index = tlx.convert_to_tensor([[0, 1, 2, 2, 3],
                                        [1, 2, 1, 0, 3]])
    x_e = tlx.gather(x, edge_index[0])
    score = segment_softmax(x_e, edge_index[1], num_segments=4)

    assert abs(sum(tlx.convert_to_numpy(score[2]) - np.array([0.7311, 0.9526]))) < 1e-4

