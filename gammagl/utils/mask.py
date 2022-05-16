# !/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2022/05/18 16:02
# @Author  : clear
# @FileName: mask.py

import tensorlayerx as tlx
from typing import Optional
import numpy as np

def index_to_mask(index, size: Optional[int] = None):
    r"""Converts indices to a mask representation.

    Args:
        index (Tensor): The indices.
        size (int, optional). The size of the mask. If set to :obj:`None`, a
            minimal sized output mask is returned.
    """
    index = tlx.convert_to_numpy(index)
    index = index.reshape((-1,))
    size = int(index.max()) + 1 if size is None else size
    mask = np.zeros(size, dtype=bool)
    mask[index] = True
    return tlx.convert_to_tensor(mask)


def mask_to_index(mask):
    r"""Converts a mask to an index representation.

    Args:
        mask (Tensor): The mask.
    """
    idx = tlx.convert_to_numpy(mask).nonzero()[0]
    return tlx.convert_to_tensor(idx, dtype=tlx.int64)
