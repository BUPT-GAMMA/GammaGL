# !/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2022/04/14 08:35
# @Author  : clear
# @FileName: tf_gpu.py.py

import os
os.environ['TL_BACKEND'] = 'tensorflow'
os.environ["CUDA_VISIBLE_DEVICES"] = "9"
import sys
sys.path.insert(0, os.path.abspath('../../'))
from pyinstrument import Profiler
import numpy as np
import tensorlayerx as tlx
import gammagl.mpops as mpops
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus :
    tf.config.experimental.set_memory_growth(gpu, True)


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
    mpops.unsorted_segment_sum(msg, dst, num_nodes)
    # mpops.unsorted_segment_mean(msg, dst, num_nodes)
    # mpops.unsorted_segment_max(msg, dst, num_nodes)
pf.stop()
print(pf.output_text(unicode=True, color=True))


dst = tlx.convert_to_numpy(dst)
idx = np.argsort(dst)
dst = tlx.gather(tlx.convert_to_tensor(dst, dtype=tlx.int64), tlx.convert_to_tensor(idx,dtype=tlx.int64))

pf.start()
for j in range(1000):
    msg = tlx.gather(x, src)
    mpops.segment_sum(msg, dst)
    # mpops.segment_mean(msg, dst)
    # mpops.segment_max(msg, dst)
pf.stop()
print(pf.output_text(unicode=True, color=True))
