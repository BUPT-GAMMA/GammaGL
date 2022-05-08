import paddle as pd

def unsorted_segment_sum(x, segment_ids, num_segments=None):
    if num_segments is not None:
        assert pd.max(segment_ids) < num_segments
    idx_ = pd.argsort(segment_ids)
    x = pd.gather(x, idx_)
    segment_ids = pd.gather(segment_ids, idx_)
    return pd.incubate.segment_sum(x, segment_ids)

def unsorted_segment_mean(x, segment_ids, num_segments=None):
    if num_segments is not None:
        assert pd.max(segment_ids) < num_segments
    idx_ = pd.argsort(segment_ids)
    x = pd.gather(x, idx_)
    segment_ids = pd.gather(segment_ids, idx_)
    return pd.incubate.segment_mean(x, segment_ids)

def unsorted_segment_max(x, segment_ids, num_segments=None):
    if num_segments is not None:
        assert pd.max(segment_ids) < num_segments
    idx_ = pd.argsort(segment_ids)
    x = pd.gather(x, idx_)
    segment_ids = pd.gather(segment_ids, idx_)
    return pd.incubate.segment_max(x, segment_ids)

def segment_sum(x, segment_ids, num_segments=None):
    if num_segments is not None:
        assert pd.max(segment_ids) < num_segments
    return pd.incubate.segment_sum(x, segment_ids)

def segment_mean(x, segment_ids, num_segments=None):
    if num_segments is not None:
        assert pd.max(segment_ids) < num_segments
    return pd.incubate.segment_mean(x, segment_ids)

def segment_max(x, segment_ids, num_segments=None):
    if num_segments is not None:
        assert pd.max(segment_ids) < num_segments
    return pd.incubate.segment_max(x, segment_ids)

