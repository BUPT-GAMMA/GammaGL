import torch
import time
import numpy as np
from torch_scatter import scatter_sum, scatter_mean, scatter_max

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# edge_index = np.load('/home/zgy/GammaGL/profiler/mpops/edge_index/cora.npy')
# edge_index = np.load('/home/zgy/GammaGL/profiler/mpops/edge_index/pubmed.npy')
edge_index = np.load('/home/zgy/GammaGL/profiler/mpops/edge_index/ogbn-arxiv.npy')

# edge_index = np.load('../../edge_index/cora.npy')
# edge_index = np.load('../../edge_index/pubmed.npy')
# edge_index = np.load('../../edge_index/ogbn-arxiv.npy')
num_nodes = np.max(edge_index)+1
src = edge_index[0,:]
dst = edge_index[1,:]
src = torch.tensor(src)
dst = torch.tensor(dst)
x = torch.tensor(np.random.randn(num_nodes, 1000))
x = x.to(device)
src = src.to(device)
dst = dst.to(device)


print("**********embedding_dim=16**********")
embedding_dim = 16
x = torch.tensor(np.random.randn(num_nodes, embedding_dim))
x = x.to(device)

msg = x[src]

start = time.time()
for j in range(10):
    scatter_sum(msg, dst, dim=0)
end = time.time()
print("scatter_sum:{:.3f}".format(end-start))

start = time.time()
for j in range(10):
    scatter_mean(msg, dst, dim=0)
end = time.time()
print("scatter_mean:{:.3f}".format(end-start))

start = time.time()
for j in range(10):
    scatter_max(msg, dst, dim=0)
end = time.time()
print("scatter_max:{:.3f}".format(end-start))

print("**********embedding_dim=16**********")


print("**********embedding_dim=64**********")
embedding_dim = 64
x = torch.tensor(np.random.randn(num_nodes, embedding_dim))
x = x.to(device)

msg = x[src]

start = time.time()
for j in range(10):
    scatter_sum(msg, dst, dim=0)
end = time.time()
print("scatter_sum:{:.3f}".format(end-start))

start = time.time()
for j in range(10):
    scatter_mean(msg, dst, dim=0)
end = time.time()
print("scatter_mean:{:.3f}".format(end-start))

start = time.time()
for j in range(10):
    scatter_max(msg, dst, dim=0)
end = time.time()
print("scatter_max:{:.3f}".format(end-start))
print("**********embedding_dim=64**********")


print("**********embedding_dim=256**********")
embedding_dim = 256
x = torch.tensor(np.random.randn(num_nodes, embedding_dim))
x = x.to(device)

msg = x[src]

start = time.time()
for j in range(10):
    scatter_sum(msg, dst, dim=0)
end = time.time()
print("scatter_sum:{:.3f}".format(end-start))

start = time.time()
for j in range(10):
    scatter_mean(msg, dst, dim=0)
end = time.time()
print("scatter_mean:{:.3f}".format(end-start))

start = time.time()
for j in range(10):
    scatter_max(msg, dst, dim=0)
end = time.time()
print("scatter_max:{:.3f}".format(end-start))
print("**********embedding_dim=256**********")

print(x.device)