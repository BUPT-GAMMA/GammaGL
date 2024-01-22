import torch
use_ext = False
try:
    from .torch_ext._torch_ext import c_segment_sum, c_segment_mean, c_segment_max, c_spmm_sum, c_spmm_mean, c_spmm_max
    use_ext = True
except:
    pass


def unsorted_segment_sum(x, segment_ids, num_segments=None):
    """
    Computes the sum along segments of a tensor.
    And :attr:`segment_ids` need not be sorted.

    Parameters
    ----------
    x : Tensor
        The raw data.
    segment_ids : Tensor
        The segment id. And the segment_ids has the same length as :attr:`x`.
    num_segments : int
        The number of segments to be divided.

    Examples
    --------
    >>> import os
    >>> os.environ['TL_BACKEND'] = 'torch'
    >>> from gammagl.mpops import unsorted_segment_sum
    >>> import tensorlayerx as tlx

    >>> x = tlx.convert_to_tensor([[1., 2., 3., 4.], [4., 3., 2., 1.], [5., 6., 7., 8.]])
    >>> segment_ids = tlx.convert_to_tensor([0, 2, 0])
    >>> num_segments = 3

    >>> unsorted_segment_sum(x, segment_ids, num_segments)
    tensor([[ 6.,  8., 10., 12.],
            [ 0.,  0.,  0.,  0.],
            [ 4.,  3.,  2.,  1.]])
    """
    assert x.shape[0] == segment_ids.shape[0], "the length of segment_ids should be equal to data.shape[0]."
    if num_segments is None:
        num_segments = int(segment_ids.max().item()) + 1
    # else:
        # `rgcn` meet an error that `segment_ids` is empty
        # assert segment_ids.max() < num_segments

    try:
        return c_segment_sum(x, segment_ids, num_segments)
    except Exception as e:
        raise e

    # if len(segment_ids.shape) == 1:
    #     s = torch.prod(torch.tensor(x.shape[1:], device=x.device)).to(torch.int64)
    #     segment_ids = segment_ids.repeat_interleave(s).view(segment_ids.shape[0], *x.shape[1:])
    #
    # assert x.shape == segment_ids.shape, "data.shape and segment_ids.shape should be equal"
    #
    # shape = [num_segments] + list(x.shape[1:])
    # tensor = torch.zeros(*shape, device=x.device).to(x.dtype).scatter_add(0, segment_ids, x)
    # return tensor


def unsorted_segment_mean(x, segment_ids, num_segments=None):
    """
    Computes the mean along segments of a tensor.
    And :attr:`segment_ids` need not be sorted.

    Parameters
    ----------
    x : Tensor
        The raw data.
    segment_ids : Tensor
        The segment id. And the segment_ids has the same length as :attr:`x`.
    num_segments : int
        The number of segments to be divided.

    Examples
    --------
    >>> import os
    >>> os.environ['TL_BACKEND'] = 'torch'
    >>> from gammagl.mpops import unsorted_segment_mean
    >>> import tensorlayerx as tlx

    >>> x = tlx.convert_to_tensor([[1., 2., 3., 4.], [4., 3., 2., 1.], [5., 6., 7., 8.]])
    >>> segment_ids = tlx.convert_to_tensor([0, 2, 0])
    >>> num_segments = 3

    >>> unsorted_segment_mean(x, segment_ids, num_segments)
    tensor([[3., 4., 5., 6.],
            [0., 0., 0., 0.],
            [4., 3., 2., 1.]])
    """
    assert x.shape[0] == segment_ids.shape[0], "the length of segment_ids should be equal to data.shape[0]."
    if num_segments is None:
        num_segments = int(segment_ids.max().item()) + 1
    # else:
    # `rgcn` meet an error that `segment_ids` is empty
    # assert segment_ids.max() < num_segments

    try:
        return c_segment_mean(x, segment_ids, num_segments)
    except Exception as e:
        raise e

    # if len(segment_ids.shape) == 1:
    #     s = torch.prod(torch.tensor(x.shape[1:], device=x.device)).to(torch.int64)
    #     segment_ids = segment_ids.repeat_interleave(s).view(segment_ids.shape[0], *x.shape[1:])
    #
    # assert x.shape == segment_ids.shape, "data.shape and segment_ids.shape should be equal"
    #
    # shape = [num_segments] + list(x.shape[1:])
    # ones_data = torch.ones_like(x, dtype=x.dtype, device=x.device)
    # tensor = torch.zeros(*shape, device=x.device).to(x.dtype).scatter_add(0, segment_ids, x)
    # tensor_nums = torch.zeros(*shape, device=x.device).to(x.dtype).scatter_add(0, segment_ids, ones_data)
    # tensor = tensor / tensor_nums
    # tensor[torch.isnan(tensor)] = 0
    # return tensor


def unsorted_segment_max(x, segment_ids, num_segments=None):
    """
    Computes the maximum along segments of a tensor.
    And :attr:`segment_ids` need not be sorted.

    Parameters
    ----------
    x : Tensor
        The raw data.
    segment_ids : Tensor
        The segment id. And the segment_ids has the same length as :attr:`x`.
    num_segments : int
        The number of segments to be divided.

    Examples
    --------
    >>> import os
    >>> os.environ['TL_BACKEND'] = 'torch'
    >>> from gammagl.mpops import unsorted_segment_max
    >>> import tensorlayerx as tlx

    >>> x = tlx.convert_to_tensor([[1., 2., 3., 4.], [4., 3., 2., 1.], [5., 6., 7., 8.]])
    >>> segment_ids = tlx.convert_to_tensor([0, 2, 0])
    >>> num_segments = 3

    >>> unsorted_segment_max(x, segment_ids, num_segments)
    tensor([[5., 6., 7., 8.],
            [0., 0., 0., 0.],
            [4., 3., 2., 1.]])
    """
    if num_segments is None:
        num_segments = int(segment_ids.max().item()) + 1
    # else:
    # `rgcn` meet an error that `segment_ids` is empty
    # assert segment_ids.max() < num_segments

    assert x.shape[0] == segment_ids.shape[0], "the length of segment_ids should be equal to data.shape[0]."
    try:
        return c_segment_max(x, segment_ids, num_segments)
    except Exception as e:
        raise e
    # res = []
    # for i in range(num_segments):
    #     res.append(torch.max(x[segment_ids == i], dim=0)[0])
    # return torch.stack(res, dim=0)


def segment_max(x, segment_ids, num_segments=None):
    """
    Computes the maximum along segments of a tensor.

    Parameters
    ----------
    x : Tensor
        The raw data.
    segment_ids : Tensor
        The segment id. And the segment_ids has the same length as :attr:`x`.
    num_segments : int
        The number of segments to be divided.

    Examples
    --------
    >>> import os
    >>> os.environ['TL_BACKEND'] = 'torch'
    >>> from gammagl.mpops import segment_max
    >>> import tensorlayerx as tlx

    >>> x = tlx.convert_to_tensor([[1., 2., 3., 4.], [4., 3., 2., 1.], [5., 6., 7., 8.]])
    >>> segment_ids = tlx.convert_to_tensor([0, 0, 1])
    >>> num_segments = 2

    >>> segment_max(x, segment_ids, num_segments)
    tensor([[4., 3., 3., 4.],
            [5., 6., 7., 8.]])
    """
    return unsorted_segment_max(x, segment_ids, num_segments)


def segment_mean(x, segment_ids, num_segments=None):
    """
    Computes the mean along segments of a tensor.

    Parameters
    ----------
    x : Tensor
        The raw data.
    segment_ids : Tensor
        The segment id. And the segment_ids has the same length as :attr:`x`.
    num_segments : int
        The number of segments to be divided.

    Examples
    --------
    >>> import os
    >>> os.environ['TL_BACKEND'] = 'torch'
    >>> from gammagl.mpops import segment_mean
    >>> import tensorlayerx as tlx

    >>> x = tlx.convert_to_tensor([[1., 2., 3., 4.], [4., 3., 2., 1.], [5., 6., 7., 8.]])
    >>> segment_ids = tlx.convert_to_tensor([0, 0, 1])
    >>> num_segments = 2

    >>> segment_mean(x, segment_ids, num_segments)
    tensor([[2.5000, 2.5000, 2.5000, 2.5000],
            [5.0000, 6.0000, 7.0000, 8.0000]])
    """
    return unsorted_segment_mean(x, segment_ids, num_segments)


def segment_sum(x, segment_ids, num_segments=None):
    """
    Computes the sum along segments of a tensor.

    Parameters
    ----------
    x : Tensor
        The raw data.
    segment_ids : Tensor
        The segment id. And the segment_ids has the same length as :attr:`x`.
    num_segments : int
        The number of segments to be divided.

    Examples
    --------
    >>> import os
    >>> os.environ['TL_BACKEND'] = 'torch'
    >>> from gammagl.mpops import segment_sum
    >>> import tensorlayerx as tlx

    >>> x = tlx.convert_to_tensor([[1., 2., 3., 4.], [4., 3., 2., 1.], [5., 6., 7., 8.]])
    >>> segment_ids = tlx.convert_to_tensor([0, 0, 1])
    >>> num_segments = 2

    >>> segment_sum(x, segment_ids, num_segments)
    tensor([[5., 5., 5., 5.],
            [5., 6., 7., 8.]])
    """
    return unsorted_segment_sum(x, segment_ids, num_segments)


def gspmm(index, weight=None, x=None, reduce='sum'):
    """
    Generalized Sparse Matrix Multiplication interface.
    It fuses two steps into one kernel: compute messages and aggregate the messages.

    Parameters
    ----------
    index: Tensor
        Edge indices of shape [2, num_edges].
    weight: Tensor, optional
        The edge weight tensor.
    x: Tensor, optional
        The node feature matrix, the dimension is [num_nodes, embedding_dim].

    Examples
    --------
    >>> import os
    >>> os.environ['TL_BACKEND'] = 'torch'
    >>> from gammagl.mpops import gspmm
    >>> import tensorlayerx as tlx
    >>> index = tlx.convert_to_tensor([[0, 1, 1, 1, 2, 3, 3, 4], [1, 0, 2, 3, 1, 1, 4, 3]])
    >>> weight = 2 * tlx.ones(shape=(tlx.get_tensor_shape(index)[1],), dtype=tlx.float32)
    >>> x = 2 * tlx.ones(shape=(max(index[0]) + 1, 8))
    >>> gspmm(index, weight, x)
    tensor([[ 4.,  4.,  4.,  4.,  4.,  4.,  4.,  4.],
        [12., 12., 12., 12., 12., 12., 12., 12.],
        [ 4.,  4.,  4.,  4.,  4.,  4.,  4.,  4.],
        [ 8.,  8.,  8.,  8.,  8.,  8.,  8.,  8.],
        [ 4.,  4.,  4.,  4.,  4.,  4.,  4.,  4.]])
    """
    if weight == None:
        weight = torch.ones(size=(index.shape[1], ), dtype=torch.float32)
    if reduce == 'sum':
        return c_spmm_sum(index, weight, x)
    elif reduce == 'mean':
        return c_spmm_mean(index, weight, x)
    elif reduce == 'max':
        return c_spmm_max(index, weight, x)
    else:
        raise Exception("Unsupported reduce type, please choose from ['sum', 'mean', 'max'].")
