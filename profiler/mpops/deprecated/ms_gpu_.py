# !/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2022/04/14 08:36
# @Author  : clear
# @FileName: ms_gpu.py

import os
os.environ['TL_BACKEND'] = 'mindspore'
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import sys
sys.path.insert(0, os.path.abspath('../../'))
from pyinstrument import Profiler
import numpy as np
import tensorlayerx as tlx
import gammagl.mpops as mpops
from mindspore.common import Tensor
from mindspore.common import dtype as mstype
from mindspore.ops import operations as P
# import mindspore.context as context
# context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')

def gather(params, indices, axis=0):
    op = P.Gather()
    return op(params, indices, axis)

def unsorted_segment_sum(x, segment_ids, num_segments):
    op = P.UnsortedSegmentSum()
    return op(x, segment_ids, num_segments)

def unsorted_segment_max(x, segment_ids, num_segments):
    op = P.UnsortedSegmentMax()
    return op(x, segment_ids, num_segments)


edge_index = np.load('edge_index.npy')
num_nodes = int(np.max(edge_index))+1
src = edge_index[0,:]
dst = edge_index[1,:]
src = tlx.convert_to_tensor(src, tlx.int32)
dst = tlx.convert_to_tensor(dst, tlx.int32)
x = tlx.convert_to_tensor(np.random.randn(num_nodes, 1000), dtype=tlx.float32)
pf = Profiler()

pf.start()
for j in range(1000):
    print(j)
    msg = tlx.gather(x, src)
    # unsorted_segment_sum(msg, dst, num_nodes)
    # mpops.unsorted_segment_mean(msg, dst, num_nodes)
    mpops.unsorted_segment_max(msg, dst, num_nodes)
pf.stop()
print(pf.output_text(unicode=True, color=True))


#####################################################


edge_index = np.load('edge_index.npy')
num_nodes = int(np.max(edge_index))+1
src = edge_index[0,:]
dst = edge_index[1,:]
src = Tensor(src, dtype=mstype.int32)
dst = Tensor(dst, dtype=mstype.int32)
x = Tensor(np.random.randn(num_nodes, 1000), dtype=mstype.float32)
pf = Profiler()




pf.start()
for j in range(100):
    msg = gather(x, src)
    # unsorted_segment_max(msg, dst, num_nodes)
    unsorted_segment_sum(msg, dst, num_nodes)
pf.stop()
print(pf.output_text(unicode=True, color=True))
