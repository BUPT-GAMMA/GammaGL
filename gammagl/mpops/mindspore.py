import mindspore as ms
from mindspore.ops import operations as P

def unsorted_segment_sum(x, segment_ids, num_segments):
    # if num_segments is not None:
    #     assert segment_ids.max() < num_segments # mindspore GPU Tensor.max 调用 reducemax，只支持float32/64
    # else:
    #     num_segments = segment_ids.max()

    op = P.UnsortedSegmentSum()
    return op(x, segment_ids, num_segments) # num_segments 只能是 int32


def unsorted_segment_mean(x, segment_ids, num_segments):
    # if num_segments is not None:
    #     assert segment_ids.max() < num_segments
    # else:
    #     num_segments = segment_ids.max()
    op = P.UnsortedSegmentSum()
    ones = ms.numpy.ones_like(x, dtype=x.dtype)
    numerator = op(x, segment_ids, num_segments)
    denominator = op(ones, segment_ids, num_segments)
    return numerator/denominator


def unsorted_segment_max(x, segment_ids, num_segments):
    # if num_segments is not None:
    #     assert segment_ids.max() < num_segments
    # else:
    #     num_segments = segment_ids.max()
    op = P.UnsortedSegmentMax()
    return op(x, segment_ids, num_segments)


def segment_sum(x, segment_ids, num_segments):
    # if num_segments is not None:
    #     assert segment_ids.max() < num_segments # mindspore GPU Tensor.max 调用 reducemax，只支持float32/64
    # else:
    #     num_segments = segment_ids.max()

    op = P.UnsortedSegmentSum()
    return op(x, segment_ids, num_segments) # num_segments 只能是 int32


def segment_mean(x, segment_ids, num_segments):
    # if num_segments is not None:
    #     assert segment_ids.max() < num_segments
    # else:
    #     num_segments = segment_ids.max()
    op = P.UnsortedSegmentSum()
    ones = ms.numpy.ones_like(x, dtype=x.dtype)
    numerator = op(x, segment_ids, num_segments)
    denominator = op(ones, segment_ids, num_segments)
    return numerator/denominator


def segment_max(x, segment_ids, num_segments):
    # if num_segments is not None:
    #     assert segment_ids.max() < num_segments
    # else:
    #     num_segments = segment_ids.max()
    op = P.UnsortedSegmentMax()
    return op(x, segment_ids, num_segments)