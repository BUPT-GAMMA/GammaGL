# -*- coding: utf-8 -*-
# @author WuJing
# @created 2023/3/28
# to cross platforms

import typing
import numpy as np
from typing import List, Tuple, Dict, Union
import tensorlayerx as tlx


# Tensorlayerx Tensor, just to take up, differentiate by TL_BACKEND.
class CommonTensor:
    @staticmethod
    def CHECK(data):
        return tlx.is_tensor(data)


Tensor = CommonTensor


def to_numpy_list(data_list):
    if isinstance(data_list, (List, Tuple)):
        data_list = list(data_list)
        for i in range(len(data_list)):
            data_list[i] = all_to_numpy(data_list[i])
        return tuple(data_list)
    return all_to_numpy(data_list)


def to_tensor_list(data_list):
    if isinstance(data_list, (List, Tuple)):
        data_list = list(data_list)
        for i in range(len(data_list)):
            data_list[i] = all_to_tensor(data_list[i])
        return tuple(data_list)
    return all_to_tensor(data_list)


# keys_type for dict-like class (__get_item__)
def all_to_numpy(data, data_to_keys: Dict[typing.Type, Union[str, List, Tuple]] = None):
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
                                data[key[0], key[1]] = all_to_numpy(data[key[0]], data_to_keys=data_to_keys)
                            except:
                                pass

                        elif len(key) == 2:
                            try:
                                data[key[0], key[1]] = all_to_numpy(data[key[0], key[1]], data_to_keys=data_to_keys)
                            except:
                                pass
                        elif len(key) == 3:
                            try:
                                data[key[0], key[1], key[2]] = all_to_numpy(data[key[0], key[1], key[2]],
                                                                            data_to_keys=data_to_keys)
                            except:
                                pass
                    try:
                        data[key] = all_to_numpy(data[key], data_to_keys=data_to_keys)
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
    else:
        return data


# convert input tensor to ndarray and convert output ndarray to tensor
def ops_func(func):
    def wrapper(*args):
        out = func(*to_numpy_list(args))
        return to_tensor_list(out)

    return wrapper
