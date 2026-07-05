from typing import Optional

import tensorlayerx as tlx

from gammagl.mpops import unsorted_segment_sum
from .check import check_is_numpy
from .num_nodes import maybe_num_nodes


def degree(index, num_nodes: Optional[int] = None, dtype=None):
    r"""Computes the (unweighted) degree of a given one-dimensional index
    tensor.

    Parameters
    ----------
    index: tensor
        Index tensor.
    num_nodes: int, optional
        The number of nodes, *i.e.*
        :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)
    dtype: :obj:`tlx.dtype`, optional
        The desired data type of the
        returned tensor.

    Returns
    -------
    :class:`Tensor`

    """
    N = maybe_num_nodes(index, num_nodes)
    if dtype is None:
        out = tlx.zeros((N, ))
    else:
        out = tlx.zeros((N,), dtype=dtype)
    one = tlx.ones((index.shape[0], ), dtype=out.dtype)
    if check_is_numpy(index):
        return tlx.unsorted_segment_sum(one, index, N)
    if hasattr(index, "device") and hasattr(one, "to"):
        one = one.to(index.device)
    return unsorted_segment_sum(one, index, N)
