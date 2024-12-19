# !/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2022/05/18 16:02
# @Author  : clear
# @FileName: mask.py


import tensorlayerx as tlx
from gammagl.utils import index_to_mask, mask_to_index


def test_index_to_mask():
    index = tlx.convert_to_tensor([1, 3, 5])

    mask = index_to_mask(index)
    assert tlx.convert_to_numpy(mask).tolist() == [False, True, False, True, False, True]

    mask = index_to_mask(index, size=7)
    assert tlx.convert_to_numpy(mask).tolist() == [False, True, False, True, False, True, False]


def test_mask_to_index():
    mask = tlx.convert_to_tensor([False, True, False, True, False, True], dtype=tlx.bool)
    index = mask_to_index(mask)
    assert tlx.convert_to_numpy(index).tolist() == [1, 3, 5]
