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


def gspmm(index, weight, x, reduce='sum'):
    """
    Generalized Sparse Matrix Multiplication interface.
    It fuses two steps into one kernel: compute messages and aggregate the messages.

    Parameters
    ----------
    index : Tensor
        Edge indices of shape [2, num_edges].
    weight : Tensor
        The edge weight tensor.
    x : Tensor
        The node feature matrix, the dimension is [num_nodex, embedding_dim].

    Examples
    --------
    >>> import os
    >>> os.environ['TL_BACKEND'] = 'torch'
    >>> from gammagl.mpops import gspmm
    >>> import tensorlayerx as tlx

    >>> index = tlx.convert_to_tensor([[0, 1, 1, 1, 2, 3, 3, 4], [1, 0, 2, 3, 1, 1, 4, 3]])
    >>> init = tlx.initializers.ones()
    >>> weight = init(shape=(tlx.get_tensor_shape(index)[1],), dtype=tlx.float32)
    >>> x = tlx.random_uniform(shape=(max(index[0]) + 1, 16))

    >>> gspmm(index, weight, x)
    tensor([[0.1123, 0.2300, 0.4464, 0.5848, 0.8047, 0.5649, 0.7504, 0.4823, 0.7156,
             0.4271, 0.6715, 0.7112, 0.4459, 0.9655, 0.2233, 0.0599],
            [2.2809, 0.8530, 1.1699, 1.6673, 1.0730, 1.2713, 2.3989, 1.3777, 1.4817,
             1.6304, 1.0347, 2.7464, 2.0170, 1.4555, 1.4429, 1.9210],
            [0.1123, 0.2300, 0.4464, 0.5848, 0.8047, 0.5649, 0.7504, 0.4823, 0.7156,
             0.4271, 0.6715, 0.7112, 0.4459, 0.9655, 0.2233, 0.0599],
            [0.2478, 0.4594, 1.0902, 0.8880, 1.6013, 1.0133, 1.5038, 1.3352, 1.6161,
             1.1601, 0.7074, 1.3606, 1.3575, 1.5835, 0.4036, 0.6374],
            [0.8745, 0.5974, 0.0056, 0.8762, 0.3908, 0.4323, 0.7849, 0.1171, 0.5689,
             0.6753, 0.0697, 0.8450, 0.7830, 0.6934, 0.4399, 0.9917]])
    """
    if reduce == 'sum':
        return c_spmm_sum(index, weight, x)
    elif reduce == 'mean':
        return c_spmm_mean(index, weight, x)
    elif reduce == 'max':
        return c_spmm_max(index, weight, x)
    else:
        raise Exception("Unsupported reduce type, please choose from ['sum', 'mean', 'max'].")
