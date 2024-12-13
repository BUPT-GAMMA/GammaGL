# -*- coding: utf-8 -*-
# @author WuJing
# @created 2023/5/31


from typing import List,Tuple
import numpy as np
import tensorlayerx as tlx
from gammagl.utils.platform_utils import all_to_tensor

if tlx.BACKEND == "paddle":
    print("Using Gammagl Lazy Load For Paddle.")

    def assert_tensor(tensor):
        if not tlx.is_tensor(tensor):
            return tlx.convert_to_tensor(tensor)
        return tensor

    origin_convert_to_tensor = tlx.convert_to_tensor
    def convert_to_tensor(value, dtype=None, device=None):
        if isinstance(value,(List,Tuple)):
            value = np.array(value)
        return origin_convert_to_tensor(value, dtype=dtype, device=device)
    tlx.convert_to_tensor = convert_to_tensor


    origin_reshape = tlx.reshape
    def reshape(tensor, shape):
        if isinstance(shape, int):
            shape = [shape]
        return origin_reshape(assert_tensor(tensor), shape)
    tlx.reshape = reshape

    origin_reduce_max = tlx.reduce_max
    def reduce_max(input_tensor, axis=None, keepdims=False):
        return origin_reduce_max(assert_tensor(input_tensor), axis, keepdims)
    tlx.reduce_max = reduce_max

    origin_reduce_min = tlx.reduce_min
    def reduce_min(input_tensor, axis=None, keepdims=False):
        return origin_reduce_min(assert_tensor(input_tensor), axis, keepdims)
    tlx.reduce_min = reduce_min

    origin_numel = tlx.numel
    def numel(input):
        if isinstance(input,np.ndarray):
            return input.size
        if isinstance(input,(List,Tuple)):
            return np.array(input).size
        return origin_numel(assert_tensor(input))
    tlx.numel = numel

    origin_multiply = tlx.multiply
    def multiply(x, y):
        return origin_multiply(assert_tensor(x), assert_tensor(y))
    tlx.multiply = multiply

    origin_gather = tlx.gather
    def gather(params, indices, axis=None):
        return origin_gather(assert_tensor(params), assert_tensor(indices), axis)
    tlx.gather = gather