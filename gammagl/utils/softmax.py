# !/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2022/04/14 25:12
# @Author  : clear
# @FileName: softmax.py
import tensorlayerx as tlx
from gammagl.mpops import *

def segment_softmax(data, segment_ids, num_segments):
    """
    segment softmax function.

    Parameters
    ----------
    data:
        The source tensor.
    segment_ids:
        The indices of elements for applying the softmax.
    num_segments:
        The number of segments.

    Returns
    -------
    tensor
        softmax score.

    """
    max_values = unsorted_segment_max(data, segment_ids, num_segments=num_segments) # tensorlayerx not supported
    gathered_max_values = tlx.gather(max_values, segment_ids)
    exp = tlx.exp(data - gathered_max_values)
    # exp = tlx.exp(data)
    denominator = unsorted_segment_sum(exp, segment_ids, num_segments=num_segments)
    gathered_denominator = tlx.gather(denominator, segment_ids)
    score = exp / (gathered_denominator + 1e-16)
    return score
