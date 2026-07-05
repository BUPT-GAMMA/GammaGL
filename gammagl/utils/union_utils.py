# coding=utf-8
import numpy as np
import tensorlayerx as tlx


def convert_union_to_numpy(data, dtype=None):
    if data is None:
        return data

    if isinstance(data, np.ndarray):
        np_data = data
    elif isinstance(data, list):
        np_data = np.array(data)
    else:
        np_data = tlx.convert_to_numpy(data)

    if dtype is not None:
        np_data = np_data.astype(dtype)

    return np_data


def union_len(data):
    shape = getattr(data, "shape", None)
    if shape is not None:
        if hasattr(shape, "as_list"):
            return shape.as_list()[0]
        return shape[0]
    return len(data)
