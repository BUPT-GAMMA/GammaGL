import torch
import tensorlayerx as tlx


def unsorted_segment_sum(x, segment_ids, num_segments):
    # segment_ids = torch.tensor(segment_ids, dtype=torch.int64)
    segment_ids = segment_ids.clone().detach()
    assert x.shape[0] == segment_ids.shape[0], "the length of segment_ids should be equal to data.shape[0]."
    if len(segment_ids.shape) == 1:
        s = torch.prod(torch.tensor(x.shape[1:], device=x.device)).to(torch.int32)
        segment_ids = segment_ids.repeat_interleave(s).view(segment_ids.shape[0], *x.shape[1:])

    assert x.shape == segment_ids.shape, "data.shape and segment_ids.shape should be equal"

    shape = [num_segments] + list(x.shape[1:])
    tensor = torch.zeros(*shape, device=x.device).to(x.dtype).scatter_add(0, segment_ids, x)
    return tensor


def degree(edge_index, num_nodes, dtype=None):
    r"""Computes the (unweighted) degree of a given one-dimensional index
    tensor.

    Args:
        index (LongTensor): Index tensor.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)
        dtype (:obj:`torch.dtype`, optional): The desired data type of the
            returned tensor.

    :rtype: :class:`Tensor`
    """
    N = num_nodes
    if dtype is None:
        out = tlx.zeros((N,), device=edge_index.device)
    else:
        out = tlx.zeros((N,), dtype=dtype, device=edge_index.device)
    one = tlx.ones((edge_index.shape[0],), dtype=out.dtype, device=out.device)
    # return tlx.unsorted_segment_sum(one,edge_index,N)
    return unsorted_segment_sum(one, edge_index, N)
