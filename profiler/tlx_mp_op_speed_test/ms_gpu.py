# !/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2022/04/14 08:36
# @Author  : clear
# @FileName: ms_gpu.py

import os
os.environ['TL_BACKEND'] = 'mindspore'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import numpy as np
import tensorlayerx as tlx


edge_index = np.load('edge_index.npy')
num_nodes = np.max(edge_index)+1
src = edge_index[0,:]
dst = edge_index[1,:]
src = tlx.convert_to_tensor(src, tlx.int64)
dst = tlx.convert_to_tensor(dst, tlx.int64)
x = tlx.convert_to_tensor(np.random.randn(num_nodes, 1000), dtype=tlx.float32)


start = time.time()
for j in range(1000):
    msg = tlx.gather(x, src)
    tlx.unsorted_segment_sum(msg, dst, num_nodes)
end = time.time()
print('mindspore unsorted spends {:.5f} s'.format(end-start))


dst = tlx.convert_to_numpy(dst)
idx = np.argsort(dst)
dst = tlx.gather(tlx.convert_to_tensor(dst, dtype=tlx.int64), tlx.convert_to_tensor(idx,dtype=tlx.int64))

start = time.time()
for j in range(1000):
    msg = tlx.gather(x, src)
    tlx.segment_sum(msg, dst)
end = time.time()
print('mindspore sorted spends {:.5f} s'.format(end-start))