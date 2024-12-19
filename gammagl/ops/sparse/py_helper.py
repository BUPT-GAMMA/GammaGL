# -*- coding: utf-8 -*-
# @author WuJing
# @created 2023/4/4

import numpy as np


def put(arr, indices, values):
    np.put(arr, indices, values)
    return arr


def new(shape, init_value=None):
    if init_value is None:
        return np.ndarray(shape)
    if isinstance(init_value, int):
        arr = np.ndarray(shape, dtype=int)
        arr.fill(init_value)
        return arr
    elif isinstance(init_value, float):
        arr = np.ndarray(shape, dtype=float)
        arr.fill(init_value)
        return

