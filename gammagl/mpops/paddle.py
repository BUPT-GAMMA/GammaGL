import paddle as pd

from paddle.fluid.layers import core
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.data_feeder import check_variable_and_dtype, check_type
from paddle.fluid.framework import in_dygraph_mode

use_ext = False
try:
    import paddle_ext
    use_ext = True
except:
    pass

def unsorted_segment_sum(x, segment_ids, num_segments=None):
    if num_segments is not None:
        assert pd.max(segment_ids) < num_segments
    else:
        num_segments = pd.max(segment_ids)+1
    if use_ext:
        return paddle_ext.unsorted_segment_sum(x, segment_ids, num_segments)
    idx_ = pd.argsort(segment_ids)
    x = pd.gather(x, idx_)
    segment_ids = pd.gather(segment_ids, idx_)
    output = pd.incubate.segment_sum(x, segment_ids)

    if output.shape[0] == num_segments:
        return output
    else:
        init_output = pd.zeros(shape=[num_segments, x.shape[1]],
                               dtype=output.dtype)
        idx = pd.arange(output.shape[0])
        final_output = _scatter(init_output, idx, output)
        return final_output


def unsorted_segment_mean(x, segment_ids, num_segments=None):
    if num_segments is not None:
        assert pd.max(segment_ids) < num_segments
    else:
        num_segments = pd.max(segment_ids)+1
    idx_ = pd.argsort(segment_ids)
    x = pd.gather(x, idx_)
    segment_ids = pd.gather(segment_ids, idx_)
    output = pd.incubate.segment_mean(x, segment_ids)

    if output.shape[0] == num_segments:
        return output
    else:
        init_output = pd.zeros(shape=[num_segments, x.shape[1]],
                               dtype=output.dtype)
        idx = pd.arange(output.shape[0])
        final_output = _scatter(init_output, idx, output)
        return final_output


def unsorted_segment_max(x, segment_ids, num_segments=None):
    if num_segments is not None:
        assert pd.max(segment_ids) < num_segments
    else:
        num_segments = pd.max(segment_ids)+1
    idx_ = pd.argsort(segment_ids)
    x = pd.gather(x, idx_)
    segment_ids = pd.gather(segment_ids, idx_)
    output = pd.incubate.segment_max(x, segment_ids)

    if output.shape[0] == num_segments:
        return output
    else:
        init_output = pd.zeros(shape=[num_segments, x.shape[1]],
                               dtype=output.dtype)
        idx = pd.arange(output.shape[0])
        final_output = _scatter(init_output, idx, output)
        return final_output


def segment_sum(x, segment_ids, num_segments=None):
    if num_segments is not None:
        assert pd.max(segment_ids) < num_segments
    else:
        num_segments = pd.max(segment_ids)+1
    output = pd.incubate.segment_sum(x, segment_ids)

    if output.shape[0] == num_segments:
        return output
    else:
        init_output = pd.zeros(shape=[num_segments, x.shape[1]],
                               dtype=output.dtype)
        idx = pd.arange(output.shape[0])
        final_output = _scatter(init_output, idx, output)
        return final_output


def segment_mean(x, segment_ids, num_segments=None):
    if num_segments is not None:
        assert pd.max(segment_ids) < num_segments
    else:
        num_segments = pd.max(segment_ids)+1
    output = pd.incubate.segment_mean(x, segment_ids)

    if output.shape[0] == num_segments:
        return output
    else:
        init_output = pd.zeros(shape=[num_segments, x.shape[1]],
                               dtype=output.dtype)
        idx = pd.arange(output.shape[0])
        final_output = _scatter(init_output, idx, output)
        return final_output


def segment_max(x, segment_ids, num_segments=None):
    if num_segments is not None:
        assert pd.max(segment_ids) < num_segments
    else:
        num_segments = pd.max(segment_ids)+1
    output = pd.incubate.segment_max(x, segment_ids)

    if output.shape[0] == num_segments:
        return output
    else:
        init_output = pd.zeros(shape=[num_segments, x.shape[1]],
                               dtype=output.dtype)
        idx = pd.arange(output.shape[0])
        final_output = _scatter(init_output, idx, output)
        return final_output


def _scatter(x, index, updates, overwrite=True):
    """
    **Scatter Layer**
    Output is obtained by updating the input on selected indices based on updates.

    .. code-block:: python

        import numpy as np
        #input:
        x = np.array([[1, 1], [2, 2], [3, 3]])
        index = np.array([2, 1, 0, 1])
        # shape of updates should be the same as x
        # shape of updates with dim > 1 should be the same as input
        updates = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
        overwrite = False
        # calculation:
        if not overwrite:
            for i in range(len(index)):
                x[index[i]] = np.zeros((2))
        for i in range(len(index)):
            if (overwrite):
                x[index[i]] = updates[i]
            else:
                x[index[i]] += updates[i]
        # output:
        out = np.array([[3, 3], [6, 6], [1, 1]])
        out.shape # [3, 2]
    **NOTICE**: The order in which updates are applied is nondeterministic,
    so the output will be nondeterministic if index contains duplicates.
    Args:
        x (Tensor): The input N-D Tensor with ndim>=1. Data type can be float32, float64.
        index (Tensor): The index 1-D Tensor. Data type can be int32, int64. The length of index cannot exceed updates's length, and the value in index cannot exceed input's length.
        updates (Tensor): update input with updates parameter based on index. shape should be the same as input, and dim value with dim > 1 should be the same as input.
        overwrite (bool): The mode that updating the output when there are same indices.
          If True, use the overwrite mode to update the output of the same index,
          if False, use the accumulate mode to update the output of the same index.Default value is True.

    Returns:
        Tensor: The output is a Tensor with the same shape as x.
    Examples:
        .. code-block:: python

            import paddle
            x = paddle.to_tensor([[1, 1], [2, 2], [3, 3]], dtype='float32')
            index = paddle.to_tensor([2, 1, 0, 1], dtype='int64')
            updates = paddle.to_tensor([[1, 1], [2, 2], [3, 3], [4, 4]], dtype='float32')

            output1 = paddle.scatter(x, index, updates, overwrite=False)
            # [[3., 3.],
            #  [6., 6.],
            #  [1., 1.]]
            output2 = paddle.scatter(x, index, updates, overwrite=True)
            # CPU device:
            # [[3., 3.],
            #  [4., 4.],
            #  [1., 1.]]
            # GPU device maybe have two results because of the repeated numbers in index
            # result 1:
            # [[3., 3.],
            #  [4., 4.],
            #  [1., 1.]]
            # result 2:
            # [[3., 3.],
            #  [2., 2.],
            #  [1., 1.]]
    """
    if in_dygraph_mode():
        return core.ops.scatter(x, index, updates, 'overwrite', overwrite)

    check_variable_and_dtype(
        x, 'dtype', ['float32', 'int32', 'int64', 'float64'], 'scatter')
    check_type(overwrite, 'overwrite', bool, 'scatter')
    helper = LayerHelper('scatter', **locals())
    out = helper.create_variable_for_type_inference(x.dtype)
    helper.append_op(
        type="scatter",
        inputs={"X": x,
                "Ids": index,
                "Updates": updates},
        attrs={'overwrite': overwrite},
        outputs={"Out": out})
    return out
