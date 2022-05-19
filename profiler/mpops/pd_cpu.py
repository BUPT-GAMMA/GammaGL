# !/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2022/04/14 08:36
# @Author  : clear
# @FileName: pd_cpu.py

import os
os.environ['TL_BACKEND'] = 'paddle'
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from pyinstrument import Profiler
import numpy as np
import tensorlayerx as tlx
from gammagl.mpops import *


edge_index = np.load('edge_index.npy')
num_nodes = np.max(edge_index)+1
src = edge_index[0,:]
dst = edge_index[1,:]
src = tlx.convert_to_tensor(src, tlx.int64)
dst = tlx.convert_to_tensor(dst, tlx.int64)
x = tlx.convert_to_tensor(np.random.randn(num_nodes, 1000), dtype=tlx.float32)
pf = Profiler()


pf.start()
for j in range(1000):
    msg = tlx.gather(x, src)
    unsorted_segment_sum(msg, dst, num_nodes)
pf.stop()
print(pf.output_text(unicode=True, color=True))


dst = tlx.convert_to_numpy(dst)
idx = np.argsort(dst)
dst = tlx.gather(tlx.convert_to_tensor(dst, dtype=tlx.int64), tlx.convert_to_tensor(idx,dtype=tlx.int64))

pf.start()
for j in range(1000):
    msg = tlx.gather(x, src)
    segment_sum(msg, dst)
pf.stop()
print(pf.output_text(unicode=True, color=True))
