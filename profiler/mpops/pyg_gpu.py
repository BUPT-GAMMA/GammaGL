# !/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2022/04/17 09:59
# @Author  : clear
# @FileName: pyg_gpu.py

import torch
import time
import numpy as np
from torch_scatter import scatter_sum, scatter_mean, scatter_max

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

edge_index = np.load('edge_index.npy')
num_nodes = np.max(edge_index)+1
src = edge_index[0,:]
dst = edge_index[1,:]
src = torch.tensor(src)
dst = torch.tensor(dst)
x = torch.tensor(np.random.randn(num_nodes, 1000))
x = x.to(device)
src = src.to(device)
dst = dst.to(device)


st = time.time()
for j in range(1000):
    msg = x[src]
    # scatter_sum(msg, dst, dim=0)
    # scatter_mean(msg, dst, dim=0)
    scatter_max(msg, dst, dim=0)
print(time.time()-st)


idx = torch.argsort(dst)
dst = dst[idx]

st = time.time()
for j in range(1000):
    msg = x[src]
    # scatter_sum(msg, dst, dim=0)
    # scatter_mean(msg, dst, dim=0)
    scatter_max(msg, dst, dim=0)
print(time.time()-st)