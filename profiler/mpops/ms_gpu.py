# !/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2022/04/14 08:36
# @Author  : clear
# @FileName: ms_gpu.py

import os
os.environ['TL_BACKEND'] = 'mindspore'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys
sys.path.insert(0, os.path.abspath('../../'))
import time
import numpy as np
import tensorlayerx as tlx
from gammagl.mpops import *

edge_index = np.load('edge_index.npy')
num_nodes = int(np.max(edge_index))+1
src = edge_index[0,:]
dst = edge_index[1,:]
src = tlx.convert_to_tensor(src, tlx.int32)
dst = tlx.convert_to_tensor(dst, tlx.int32)
msg = tlx.convert_to_tensor(np.random.randn(edge_index.shape[1], 500), dtype=tlx.float32)


start_t = time.time()
for j in range(200):
    # msg = tlx.gather(x, src)
    # unsorted_segment_sum(msg, dst, num_nodes)
    # unsorted_segment_mean(msg, dst, num_nodes)
    unsorted_segment_max(msg, dst, num_nodes)
print("{:.3f}".format(time.time()-start_t))
# pf.stop()
# print(pf.output_text(unicode=True, color=True))


dst = tlx.convert_to_numpy(dst)
idx = np.argsort(dst)
dst = tlx.gather(tlx.convert_to_tensor(dst, dtype=tlx.int32), tlx.convert_to_tensor(idx,dtype=tlx.int32))

# pf.start()
start_t = time.time()
for j in range(200):
    # msg = tlx.gather(x, src)
    # segment_sum(msg, dst, num_nodes)
    # segment_mean(msg, dst, num_nodes)
    segment_max(msg, dst, num_nodes)
print("{:.3f}".format(time.time()-start_t))
# pf.stop()
# print(pf.output_text(unicode=True, color=True))