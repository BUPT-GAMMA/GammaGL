# -*- coding: utf-8 -*-
# @author WuJing
# @created 2023/3/28
# to cross platforms

import typing
from time import time

import numpy as np
from typing import List, Tuple, Dict, Union
import tensorlayerx as tlx


# Tensorlayerx Tensor, just to take up, differentiate by TL_BACKEND.
class CommonTensor:
    @staticmethod
    def CHECK(data):
        return tlx.is_tensor(data)


Tensor = CommonTensor

# typing

NodeType = str
EdgeType = Tuple[str, str, str]


# Adj = Union[Tensor, SparseGraph]
# OptTensor = typing.Optional[Tensor]
# InputNodes = Union[OptTensor, NodeType, Tuple[NodeType, OptTensor]]
# InputEdges = Union[OptTensor, EdgeType, Tuple[EdgeType, OptTensor]]
# NumNeighbors = Union[List[int], Dict[EdgeType, List[int]]]


# change all tensor data dtype to int64
def with_dtype(data: Union[np.ndarray, Tensor], dtype):
    if isinstance(data, np.ndarray):
        return data.astype(dtype)
    elif not Tensor.CHECK(data):
        raise TypeError("Only ndarray and Tensor type could be cast dtype!")
    else:
        return tlx.cast(data, dtype)


def as_int64(data: Union[np.ndarray, Tensor]):
    return with_dtype(data, tlx.int64)


def as_int32(data: Union[np.ndarray, Tensor]):
    return with_dtype(data, tlx.int32)


def as_float32(data: Union[np.ndarray, Tensor]):
    return with_dtype(data, tlx.float32)


def as_float64(data: Union[np.ndarray, Tensor]):
    return with_dtype(data, tlx.float64)


as_double = as_float64

as_long = as_int64


def to_numpy_list(data_list: List):
    if isinstance(data_list, List):
        for i in range(len(data_list)):
            data_list[i] = all_to_numpy(data_list[i])
    return data_list


# each element to tensor
def to_tensor_list(data_list: List):
    if isinstance(data_list, Tuple):
        data_list = list(data_list)
    if isinstance(data_list, (List, np.ndarray)):
        for i in range(len(data_list)):
            data_list[i] = all_to_tensor(data_list[i])
    return data_list


# arr-like data to list
def to_list(data: Union[np.ndarray, Tensor]):
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif Tensor.CHECK(data):
        return all_to_numpy(data).tolist()
    else:
        return list(data)

# keys_type for dict-like class (__get_item__)
def all_to_numpy(data):
    if data is None:
        return data
    elif isinstance(data, np.ndarray):
        return data
    elif isinstance(data, (List, Tuple)):
        return np.array(data)
    elif tlx.is_tensor(data):
        return tlx.convert_to_numpy(data)
    elif isinstance(data, Dict):
        for k in data:
            if tlx.is_tensor(data[k]):
                data[k] = tlx.convert_to_numpy(data[k])
        return data
    elif hasattr(data, '__class__') and hasattr(data.__class__, '__iter__'):
        for k, v in data:
            data[k] = all_to_numpy(v)
        return data
    else:
        return data

    return data

def all_to_numpy_by_dict(data, data_to_keys: Dict[typing.Type, Union[str, List, Tuple]] = None):
    if data is None:
        return data
    elif isinstance(data, np.ndarray):
        return data
    elif isinstance(data, (List, Tuple)):
        return np.array(data)
    elif tlx.is_tensor(data):
        return tlx.convert_to_numpy(data)
    elif isinstance(data, Dict):
        for k in data:
            if tlx.is_tensor(data[k]):
                data[k] = tlx.convert_to_numpy(data[k])
        return data
    elif data_to_keys is not None:
        for keys_type in data_to_keys.keys():
            if isinstance(data, keys_type):
                keys = data_to_keys[keys_type]
                if not isinstance(data_to_keys[keys_type], (List, Tuple)):
                    keys = [data_to_keys[keys_type]]
                for key in keys:
                    if isinstance(key, (List, Tuple)):
                        # support hetero graph class
                        if len(key) == 0:
                            continue
                        elif len(key) == 1:
                            try:
                                data[key[0], key[1]] = all_to_numpy_by_dict(data[key[0]], data_to_keys=data_to_keys)
                            except:
                                pass

                        elif len(key) == 2:
                            try:
                                data[key[0], key[1]] = all_to_numpy_by_dict(data[key[0], key[1]], data_to_keys=data_to_keys)
                            except:
                                pass
                        elif len(key) == 3:
                            try:
                                data[key[0], key[1], key[2]] = all_to_numpy_by_dict(data[key[0], key[1], key[2]],
                                                                            data_to_keys=data_to_keys)
                            except:
                                pass
                    try:
                        data[key] = all_to_numpy_by_dict(data[key], data_to_keys=data_to_keys)
                    except:
                        pass
        return data
    elif hasattr(data, '__class__') and hasattr(data.__class__, '__iter__'):
        for k, v in data:
            data[k] = all_to_numpy(v)
        return data
    else:
        return data

    return data


def all_to_tensor(data, dtype=tlx.int64):
    if data is None:
        return data
    elif tlx.is_tensor(data):
        return data
    elif isinstance(data, (List, Tuple, np.ndarray)):
        return tlx.convert_to_tensor(data, dtype=dtype)
    elif isinstance(data, Dict):
        for k in data:
            data[k] = all_to_tensor(data[k])
        return data
    elif hasattr(data, '__class__') and hasattr(data.__class__, '__iter__'):
        for k, v in data:
            data[k] = all_to_tensor(v)
        return data
    else:
        return data


# convert input tensor to ndarray and convert output ndarray to tensor
def out_tensor(func):
    def wrapper(*args):
        return all_to_tensor(func(*args))
    return wrapper


# convert input tensor to ndarray and convert output ndarray to tensor
def out_tensor_list(func):
    def wrapper(*args):
        return to_tensor_list(func(*args))

    return wrapper
