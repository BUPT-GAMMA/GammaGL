# !/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2022/04/21 06:23
# @Author  : clear
# @FileName: msgpass_speed_test.py.py

import os
os.environ['TL_BACKEND'] = 'torch' # set your backend here, default `tensorflow`
# os.environ["CUDA_VISIBLE_DEVICES"] = "8"
import sys
sys.path.insert(0, os.path.abspath('../../')) # adds path2gammagl to execute in command line.
import tensorlayerx as tlx
from gammagl.datasets import Planetoid
from gammagl.utils.loop import add_self_loops
import time
import numpy as np


dataset = Planetoid(r'./', 'pubmed')
dataset.process()
graph = dataset[0]
graph.tensor()
edge_index, _ = add_self_loops(graph.edge_index, n_loops=1, num_nodes=19717)
src = edge_index[0,:]
dst = edge_index[1,:]

# dst = tlx.convert_to_numpy(dst)
# idx = np.argsort(dst)
# dst = tlx.gather(tlx.convert_to_tensor(dst, dtype=tlx.int64), tlx.convert_to_tensor(idx,dtype=tlx.int64))


import torch
from torch_scatter import scatter_sum
device = torch.device("cuda:8" if torch.cuda.is_available() else "cpu")
src=src.to(device)
dst = dst.to(device)
total_time = 0
for i in range(100):
    x = torch.tensor(np.random.randn(19717, 500), dtype=tlx.float32)
    x = x.to(device)
    start_time = time.time()
    for j in range(100):
        msg = x[src]
        scatter_sum(msg, dst,dim=0)
    end_time = time.time()
    total_time += end_time-start_time

print('torch_scatter spends {:.5f}s'.format(total_time))
#
#
# device = torch.device("cuda:8" if torch.cuda.is_available() else "cpu")
# dst = dst.to(device)
# src= src.to(device)
# def unsorted_segment_sum(z, x, dst, num_segments):
#     tensor = z.scatter_add(0, dst, x)
#     return tensor
# def segment_sum(x, dst, num_segments):
#     # dst = torch.tensor(dst, dtype=torch.int64,device=device)
#     # num_segments = len(torch.unique(dst))
#     return unsorted_segment_sum(x, dst, num_segments)
#
# # 真实连边，虚假feature，并重置
# total_time = 0
# for i in range(100):
#     x = tlx.convert_to_tensor(np.random.randn(19717, 500), dtype=tlx.float32)
#     x = x.to(device)
#     shape = [19717] + list(x.shape[1:])
#     z = torch.zeros(*shape, device=device).to(x.dtype)
#     if len(dst.shape) == 1:
#         t = torch.tensor(x.shape[1:]).to(device)
#         s = torch.prod(t).to(torch.int64)
#         dst = dst.repeat_interleave(s).view(dst.shape[0], *x.shape[1:])
#     start_time = time.time()
#     for j in range(100):
#         msg = tlx.gather(x, src, axis=0)
#         # unsorted_segment_sum(z, msg, dst, 19717)
#         segment_sum(msg, dst)
#     end_time = time.time()
#     total_time += end_time-start_time
#
# print(os.environ['TL_BACKEND']+' spends {:.5f}s'.format(total_time))
