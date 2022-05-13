# !/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2022/04/14 25:12
# @Author  : clear
# @FileName: softmax.py
import tensorlayerx as tlx

def segment_softmax(data, segment_ids, num_segments):
    # max_values = tlx.unsorted_segment_max(data, segment_ids, num_segments=num_segments) # tensorlayerx not supported
    # gathered_max_values = tlx.gather(max_values, segment_ids)
    # exp = tlx.exp(data - tf.stop_gradient(gathered_max_values))
    exp = tlx.exp(data) # - gathered_max_values)
    denominator = tlx.unsorted_segment_sum(exp, segment_ids, num_segments=num_segments) + 1e-8
    gathered_denominator = tlx.gather(denominator, segment_ids)
    score = exp / (gathered_denominator + 1e-16)
    return score