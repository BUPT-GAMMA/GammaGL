import mindspore as ms
from mindspore.ops import operations as P

def unsorted_segment_sum(x, segment_ids, num_segments=None):
    if num_segments is not None:
        assert segment_ids.max() < num_segments
    else:
        num_segments = segment_ids.max()
    op = P.UnsortedSegmentSum()
    return op(x, segment_ids, num_segments)


def unsorted_segment_mean(x, segment_ids, num_segments=None):
    if num_segments is not None:
        assert segment_ids.max() < num_segments
    else:
        num_segments = segment_ids.max()
    op = P.UnsortedSegmentSum()
    ones = ms.numpy.ones_like(x, dtype=x.dtype)
    numerator = op(x, segment_ids, num_segments)
    denominator = op(ones, segment_ids, num_segments)
    return numerator/denominator


def unsorted_segment_max(x, segment_ids, num_segments=None):
    if num_segments is not None:
        assert segment_ids.max() < num_segments
    else:
        num_segments = segment_ids.max()
    op = P.UnsortedSegmentMax()
    return op(x, segment_ids, num_segments)