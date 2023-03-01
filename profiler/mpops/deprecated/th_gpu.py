# !/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2022/04/14 08:35
# @Author  : clear
# @FileName: th_gpu.py

import os
os.environ['TL_BACKEND'] = 'torch'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
sys.path.insert(0, os.path.abspath('../../'))
import torch
# from pyinstrument import Profiler
import numpy as np
import tensorlayerx as tlx
from gammagl.mpops import *
import time


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

edge_index = np.load('edge_index.npy')
num_nodes = np.max(edge_index)+1
src = edge_index[0,:]
dst = edge_index[1,:]
src = tlx.convert_to_tensor(src, tlx.int64)
dst = tlx.convert_to_tensor(dst, tlx.int64)
msg = tlx.convert_to_tensor(np.random.randn(edge_index.shape[1], 500), dtype=tlx.float32)
msg = msg.to(device)
src = src.to(device)
dst = dst.to(device)
# pf = Profiler()

# pf.start()
start_t = time.time()
for j in range(200):
    # unsorted_segment_sum(msg, dst, num_nodes)
    # unsorted_segment_mean(msg, dst, num_nodes)
    unsorted_segment_max(msg, dst, num_nodes)
print("{:.3f}".format(time.time()-start_t))
# pf.stop()
# print(pf.output_text(unicode=True, color=True))



idx = torch.argsort(dst)
dst = tlx.gather(dst, idx)
# dst = dst.to(device)

# pf.start()
start_t = time.time()
for j in range(200):
    # segment_sum(msg, dst, num_nodes)
    # segment_mean(msg, dst, num_nodes)
    segment_max(msg, dst, num_nodes)
print("{:.3f}".format(time.time()-start_t))
# pf.stop()
# print(pf.output_text(unicode=True, color=True))
