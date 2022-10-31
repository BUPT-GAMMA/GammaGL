import numpy as np
import paddle
import tensorlayerx as tlx
import tensorflow as tf

TF_BACKEND = 'tensorflow'
TORCH_BACKEND = 'torch'
PADDLE_BACKEND = 'paddle'
MS_BACKEND = 'mindspore'

def split_to_two(tensor, axis=-1):
    if tlx.BACKEND == TORCH_BACKEND or tlx.BACKEND == MS_BACKEND:
        return tlx.split(tensor, tensor.shape[-1] // 2, axis)
    elif tlx.BACKEND == TF_BACKEND:
        return tlx.split(tensor, 2, axis)
    elif tlx.BACKEND == PADDLE_BACKEND:
        return tensor.split(2, axis=axis)


def only_hot_to_idx(y):
    res = np.ndarray([y.shape[0], 1], np.int32)
    for k, e in enumerate(y):
        for k2, e2 in enumerate(e):
            if e2 == 1:
                break
        res[k] = k2
    return res


def cast(tensor, dtype):
    if tlx.BACKEND == TORCH_BACKEND:
        return tensor.int() if dtype == tlx.int32 else tensor.float() if dtype == tlx.float32 else tensor.double if dtype == tlx.float64 else tensor.long() if dtype == tlx.int64 else tensor
    elif tlx.BACKEND == TF_BACKEND:
        return tf.cast(tensor,
                       dtype=tf.int32 if dtype == tlx.int32 else tf.float32 if dtype == tlx.float32 else tf.float64 if dtype == tlx.float64 else tf.int64 if dtype == tlx.int64 else tensor.dtype)
    elif tlx.BACKEND == PADDLE_BACKEND:
        return tensor.astype(dtype=dtype)
    else:
        return None

def product(xs, ys):
    res = []
    for x in xs:
        for y in ys:
            res.append((x, y))
    return res