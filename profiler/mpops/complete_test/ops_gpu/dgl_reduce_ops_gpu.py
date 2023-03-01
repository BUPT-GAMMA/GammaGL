import torch
import dgl.ops as F
import dgl
import numpy as np
import time

device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

edge_index = np.load('/home/zgy/GammaGL/profiler/mpops/edge_index/cora.npy')
# edge_index = np.load('/home/zgy/GammaGL/profiler/mpops/edge_index/pubmed.npy')
# edge_index = np.load('/home/zgy/GammaGL/profiler/mpops/edge_index/ogbn-arxiv.npy')

# edge_index = np.load('../../edge_index/cora.npy')
# edge_index = np.load('../../edge_index/pubmed.npy')
# edge_index = np.load('../../edge_index/ogbn-arxiv.npy')
num_nodes = np.max(edge_index) + 1
src = edge_index[0, :]
dst = edge_index[1, :]

g = dgl.graph((edge_index[0], edge_index[1]))
g = g.to(device)
# g.ndata['h'] = x
# message_func = dgl.function.copy_u('h', 'm')
# x = torch.tensor(np.random.randn(g.num_nodes(), 16))
#
# out = F.copy_u_sum(g, x)
# print(out)
it = 10000

print("**********embedding_dim=16**********")
embedding_dim = 16
x = torch.tensor(np.random.randn(num_nodes, embedding_dim))
x = x.to(device)

g.ndata['h'] = x
message_func = dgl.function.copy_u('h', 'm')

start = time.time()
for j in range(it):
    dgl.function.sum('m', 'h')
end = time.time()
print("copy_u_sum:{:.3f}".format(end - start))

start = time.time()
for j in range(it):
    dgl.function.mean('m', 'h')
end = time.time()
print("copy_u_mean:{:.3f}".format(end - start))

start = time.time()
for j in range(it):
    dgl.function.max('m', 'h')
end = time.time()
print("copy_u_max:{:.3f}".format(end - start))

print("**********embedding_dim=16**********")

print("**********embedding_dim=64**********")
embedding_dim = 64
x = torch.tensor(np.random.randn(num_nodes, embedding_dim))
x = x.to(device)

g.ndata['h'] = x
message_func = dgl.function.copy_u('h', 'm')

start = time.time()
for j in range(it):
    dgl.function.sum('m', 'h')
end = time.time()
print("copy_u_sum:{:.3f}".format(end - start))

start = time.time()
for j in range(it):
    dgl.function.mean('m', 'h')
end = time.time()
print("copy_u_mean:{:.3f}".format(end - start))

start = time.time()
for j in range(it):
    dgl.function.max('m', 'h')
end = time.time()
print("copy_u_max:{:.3f}".format(end - start))

print("**********embedding_dim=64**********")

print("**********embedding_dim=256**********")
embedding_dim = 256
x = torch.tensor(np.random.randn(num_nodes, embedding_dim))
x = x.to(device)

g.ndata['h'] = x
message_func = dgl.function.copy_u('h', 'm')

start = time.time()
for j in range(it):
    dgl.function.sum('m', 'h')
end = time.time()
print("copy_u_sum:{:.3f}".format(end - start))

start = time.time()
for j in range(it):
    dgl.function.mean('m', 'h')
end = time.time()
print("copy_u_mean:{:.3f}".format(end - start))

start = time.time()
for j in range(it):
    dgl.function.max('m', 'h')
end = time.time()
print("copy_u_max:{:.3f}".format(end - start))

print("**********embedding_dim=256**********")
print(x.device)
