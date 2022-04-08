# !/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2022/04/14 08:35
# @Author  : clear
# @FileName: th_gpu.py

import os
os.environ['TL_BACKEND'] = 'torch'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from pyinstrument import Profiler
import numpy as np
import tensorlayerx as tlx

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def unsorted_segment_sum(x, segment_ids, num_segments):

    segment_ids = torch.tensor(segment_ids, dtype=torch.int64)
    assert x.shape[0] == segment_ids.shape[0], "the length of segment_ids should be equal to data.shape[0]."
    if len(segment_ids.shape) == 1:
        s = torch.prod(torch.tensor(x.shape[1:])).to(torch.int32).item()
        segment_ids = segment_ids.repeat_interleave(s).view(segment_ids.shape[0], *x.shape[1:])

    assert x.shape == segment_ids.shape, "data.shape and segment_ids.shape should be equal"

    shape = [num_segments] + list(x.shape[1:])
    tensor = torch.zeros(*shape, dtype=x.dtype,device=device).scatter_add(0, segment_ids, x)
    return tensor



edge_index = np.load('edge_index.npy')
num_nodes = np.max(edge_index)+1
src = edge_index[0,:]
dst = edge_index[1,:]
src = tlx.convert_to_tensor(src, tlx.int64)
dst = tlx.convert_to_tensor(dst, tlx.int64)
x = tlx.convert_to_tensor(np.random.randn(num_nodes, 1000), dtype=tlx.float32)
x = x.to(device)
src = src.to(device)
dst = dst.to(device)
pf = Profiler()

pf.start()
for j in range(1000):
    msg = tlx.gather(x, src)
    unsorted_segment_sum(msg, dst, num_nodes)
pf.stop()
print(pf.output_text(unicode=True, color=True))



idx = torch.argsort(dst)
dst = tlx.gather(dst, idx)
# dst = dst.to(device)

pf.start()
for j in range(1000):
    msg = tlx.gather(x, src)
    unsorted_segment_sum(msg, dst, num_nodes)
pf.stop()
print(pf.output_text(unicode=True, color=True))
