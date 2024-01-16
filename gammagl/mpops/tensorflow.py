import tensorflow as tf


def unsorted_segment_sum(x, segment_ids, num_segments=None):
    if num_segments is not None:
        assert tf.reduce_max(segment_ids) < num_segments
    else:
        num_segments = tf.reduce_max(segment_ids)+1
        
    return tf.math.unsorted_segment_sum(x, segment_ids, num_segments)


def unsorted_segment_mean(x, segment_ids, num_segments=None):
    if num_segments is not None:
        assert tf.reduce_max(segment_ids) < num_segments
    else:
        num_segments = tf.reduce_max(segment_ids)+1
    return tf.math.unsorted_segment_mean(x, segment_ids, num_segments)


def unsorted_segment_max(x, segment_ids, num_segments=None):
    if num_segments is not None:
        assert tf.reduce_max(segment_ids) < num_segments
    else:
        num_segments = tf.reduce_max(segment_ids)+1
    return tf.math.unsorted_segment_max(x, segment_ids, num_segments)


def unsorted_segment_min(x, segment_ids, num_segments=None):
    if num_segments is not None:
        assert tf.reduce_max(segment_ids) < num_segments
    else:
        num_segments = tf.reduce_max(segment_ids)+1
    return tf.math.unsorted_segment_min(x, segment_ids, num_segments)


def segment_sum(x, segment_ids, num_segments=None):
    if num_segments is not None:
        assert tf.reduce_max(segment_ids) < num_segments
    return tf.math.segment_sum(x, segment_ids)


def segment_mean(x, segment_ids, num_segments=None):
    if num_segments is not None:
        assert tf.reduce_max(segment_ids) < num_segments
    return tf.math.segment_mean(x, segment_ids)


def segment_max(x, segment_ids, num_segments=None):
    if num_segments is not None:
        assert tf.reduce_max(segment_ids) < num_segments
    return tf.math.segment_max(x, segment_ids)


def segment_min(x, segment_ids, num_segments=None):
    if num_segments is not None:
        assert tf.reduce_max(segment_ids) < num_segments
    return tf.math.segment_min(x, segment_ids)


def gspmm(index, weight=None, x=None, reduce='sum'):
    pass
