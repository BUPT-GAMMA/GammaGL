import mindspore as ms
from mindspore.ops import operations as P

def unsorted_segment_sum(x, segment_ids, num_segments=None):
    if num_segments is None:
        num_segments = int(segment_ids.asnumpy().max() + 1)
    # else:
    #     assert segment_ids.max() < num_segments # mindspore GPU Tensor.max 调用 reducemax，只支持float32/64

    op = P.UnsortedSegmentSum()
    return op(x, segment_ids, num_segments) # num_segments 只能是 int32


def unsorted_segment_mean(x, segment_ids, num_segments=None):
    if num_segments is None:
        num_segments = int(segment_ids.asnumpy().max() + 1)
    # else:
    #     assert segment_ids.max() < num_segments

    op = P.UnsortedSegmentSum()
    ones = ms.numpy.ones_like(x, dtype=x.dtype)
    numerator = op(x, segment_ids, num_segments)
    denominator = op(ones, segment_ids, num_segments)
    return numerator/denominator


def unsorted_segment_max(x, segment_ids, num_segments=None):
    if num_segments is None:
        num_segments = int(segment_ids.asnumpy().max() + 1)
    # else:
    #     assert segment_ids.max() < num_segments

    op = P.UnsortedSegmentMax()
    return op(x, segment_ids, num_segments)


def segment_sum(x, segment_ids, num_segments=None):
    if num_segments is None:
        num_segments = int(segment_ids.asnumpy().max() + 1)
    op = P.UnsortedSegmentSum()
    return op(x, segment_ids, num_segments) # num_segments 只能是 int32


def segment_mean(x, segment_ids, num_segments=None):
    if num_segments is None:
        num_segments = int(segment_ids.asnumpy().max() + 1)
    op = P.UnsortedSegmentSum()
    ones = ms.numpy.ones_like(x, dtype=x.dtype)
    numerator = op(x, segment_ids, num_segments)
    denominator = op(ones, segment_ids, num_segments)
    return numerator/denominator


def segment_max(x, segment_ids, num_segments=None):
    if num_segments is None:
        num_segments = int(segment_ids.asnumpy().max() + 1)
    op = P.UnsortedSegmentMax()
    return op(x, segment_ids, num_segments)