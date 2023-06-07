import os

os.environ['TL_BACKEND'] = 'torch'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from pyinstrument import Profiler
import numpy as np
import tensorlayerx as tlx
# from gammagl.mpops import *
import time

try:
    import torch_segment
except ImportError:
    exit(0)

relative_path = 'profiler/mpops/edge_index/'
file_name = ['cora.npy', 'pubmed.npy', 'ogbn-arxiv.npy']
embedding = [16, 64, 256]
iter = 10

for name in file_name:
    path = relative_path + name
    print(path)
    edge_index = np.load(path)

    num_nodes = np.max(edge_index) + 1
    src = edge_index[0, :]
    dst = edge_index[1, :]
    src = tlx.convert_to_tensor(src, tlx.int64)
    dst = tlx.convert_to_tensor(dst, tlx.int64)
    # x = tlx.convert_to_tensor(np.random.randn(num_nodes, embedding_dim), dtype=tlx.float32)
    edge_index = tlx.convert_to_tensor(edge_index)

    for embedding_dim in embedding:

        print("**********embedding_dim={}**********".format(embedding_dim))
        x = tlx.convert_to_tensor(np.random.randn(num_nodes, embedding_dim), dtype=tlx.float32)
        msg = tlx.gather(x, src)

        start = time.time()
        for j in range(iter):
            torch_segment.segment_max(msg, dst, num_nodes)
        end = time.time()
        print("ext_segment_max:{:.3f}".format(end-start))

        print("**********embedding_dim={}**********".format(embedding_dim))

print(x.device)
