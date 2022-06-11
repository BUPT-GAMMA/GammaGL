# !/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2022/05/00 16:47
# @Author  : clear
# @FileName: test_get_laplacian.py
import tensorlayerx as tlx
from gammagl.utils.get_laplacian import get_laplacian


def test_get_laplacian():
    edge_index = tlx.convert_to_tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=tlx.int64)
    edge_weight = tlx.convert_to_tensor([1, 2, 2, 4], dtype=tlx.float32)

    lap = get_laplacian(edge_index, 3, edge_weight)
    assert tlx.convert_to_numpy(lap[0]).tolist() == [[0, 1, 1, 2, 0, 1, 2], [1, 0, 2, 1, 0, 1, 2]]
    assert tlx.convert_to_numpy(lap[1]).tolist() == [-1, -2, -2, -4, 1, 4, 4]

    lap_sym = get_laplacian(edge_index, 3,  edge_weight, normalization='sym')
    assert tlx.convert_to_numpy(lap_sym[0]).tolist() == tlx.convert_to_numpy(lap[0]).tolist()
    assert tlx.convert_to_numpy(lap_sym[1]).tolist() == [-0.5, -1, -0.5, -1, 1, 1, 1]

    lap_rw = get_laplacian(edge_index, 3, edge_weight, normalization='rw')
    assert tlx.convert_to_numpy(lap_rw[0]).tolist() == tlx.convert_to_numpy(lap[0]).tolist()
    assert tlx.convert_to_numpy(lap_rw[1]).tolist() == [-1, -0.5, -0.5, -1, 1, 1, 1]
