import os

os.environ['TL_BACKEND'] = 'tensorflow'
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
from pyinstrument import Profiler
import numpy as np
import tensorlayerx as tlx
from gammagl.mpops import *
import time


# edge_index = np.load('/home/zgy/GammaGL/profiler/mpops/edge_index/cora.npy')
# edge_index = np.load('/home/zgy/GammaGL/profiler/mpops/edge_index/pubmed.npy')
edge_index = np.load('/home/zgy/GammaGL/profiler/mpops/edge_index/ogbn-arxiv.npy')

# edge_index = np.load('../../edge_index/cora.npy')
# edge_index = np.load('../../edge_index/pubmed.npy')
# edge_index = np.load('../../edge_index/ogbn-arxiv.npy')
num_nodes = np.max(edge_index) + 1
src = edge_index[0, :]
dst = edge_index[1, :]
src = tlx.convert_to_tensor(src, tlx.int64)
dst = tlx.convert_to_tensor(dst, tlx.int64)
# x = tlx.convert_to_tensor(np.random.randn(num_nodes, embedding_dim), dtype=tlx.float32)
edge_index = tlx.convert_to_tensor(edge_index)

dst2 = tlx.convert_to_numpy(dst)
idx = np.argsort(dst2)
dst2 = tlx.gather(tlx.convert_to_tensor(dst2, dtype=tlx.int64), tlx.convert_to_tensor(idx,dtype=tlx.int64))


print("**********embedding_dim=16**********")
embedding_dim = 16
x = tlx.convert_to_tensor(np.random.randn(num_nodes, embedding_dim), dtype=tlx.float32)
msg = tlx.gather(x, src)

start = time.time()
for j in range(10):
    unsorted_segment_sum(msg, dst, num_nodes)
end = time.time()
print("unsorted_segment_sum:{:.3f}".format(end-start))

start = time.time()
for j in range(10):
    segment_sum(msg, dst2, num_nodes)
end = time.time()
print("segment_sum:{:.3f}".format(end-start))

start = time.time()
for j in range(10):
    unsorted_segment_mean(msg, dst, num_nodes)
end = time.time()
print("unsorted_segment_mean:{:.3f}".format(end-start))

start = time.time()
for j in range(10):
    segment_mean(msg, dst2, num_nodes)
end = time.time()
print("segment_mean:{:.3f}".format(end-start))

start = time.time()
for j in range(10):
    unsorted_segment_max(msg, dst, num_nodes)
end = time.time()
print("unsorted_segment_max:{:.3f}".format(end-start))

start = time.time()
for j in range(10):
    segment_max(msg, dst2, num_nodes)
end = time.time()
print("segment_max:{:.3f}".format(end-start))

print("**********embedding_dim=16**********")


print("**********embedding_dim=64**********")
embedding_dim = 64
x = tlx.convert_to_tensor(np.random.randn(num_nodes, embedding_dim), dtype=tlx.float32)
msg = tlx.gather(x, src)

start = time.time()
for j in range(10):
    unsorted_segment_sum(msg, dst, num_nodes)
end = time.time()
print("unsorted_segment_sum:{:.3f}".format(end-start))

start = time.time()
for j in range(10):
    segment_sum(msg, dst2, num_nodes)
end = time.time()
print("segment_sum:{:.3f}".format(end-start))

start = time.time()
for j in range(10):
    unsorted_segment_mean(msg, dst, num_nodes)
end = time.time()
print("unsorted_segment_mean:{:.3f}".format(end-start))

start = time.time()
for j in range(10):
    segment_mean(msg, dst2, num_nodes)
end = time.time()
print("segment_mean:{:.3f}".format(end-start))

start = time.time()
for j in range(10):
    unsorted_segment_max(msg, dst, num_nodes)
end = time.time()
print("unsorted_segment_max:{:.3f}".format(end-start))

start = time.time()
for j in range(10):
    segment_max(msg, dst2, num_nodes)
end = time.time()
print("segment_max:{:.3f}".format(end-start))

print("**********embedding_dim=64**********")


print("**********embedding_dim=256**********")
embedding_dim = 256
x = tlx.convert_to_tensor(np.random.randn(num_nodes, embedding_dim), dtype=tlx.float32)
msg = tlx.gather(x, src)

start = time.time()
for j in range(10):
    unsorted_segment_sum(msg, dst, num_nodes)
end = time.time()
print("unsorted_segment_sum:{:.3f}".format(end-start))

start = time.time()
for j in range(10):
    segment_sum(msg, dst2, num_nodes)
end = time.time()
print("segment_sum:{:.3f}".format(end-start))

start = time.time()
for j in range(10):
    unsorted_segment_mean(msg, dst, num_nodes)
end = time.time()
print("unsorted_segment_mean:{:.3f}".format(end-start))

start = time.time()
for j in range(10):
    segment_mean(msg, dst2, num_nodes)
end = time.time()
print("segment_mean:{:.3f}".format(end-start))

start = time.time()
for j in range(10):
    unsorted_segment_max(msg, dst, num_nodes)
end = time.time()
print("unsorted_segment_max:{:.3f}".format(end-start))

start = time.time()
for j in range(10):
    segment_max(msg, dst2, num_nodes)
end = time.time()
print("segment_max:{:.3f}".format(end-start))

print("**********embedding_dim=256**********")

print(x.device)
