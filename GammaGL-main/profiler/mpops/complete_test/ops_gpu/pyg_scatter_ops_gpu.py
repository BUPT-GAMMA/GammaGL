import torch
import time
import numpy as np
from torch_scatter import scatter_sum, scatter_mean, scatter_max

device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

relative_path = 'profiler/mpops/edge_index/'
file_name = ['cora.npy', 'pubmed.npy', 'ogbn-arxiv.npy']
embedding = [16, 64, 256]
iter = 10

for name in file_name:
    path = relative_path + name
    print(path)
    edge_index = np.load(path)

    num_nodes = np.max(edge_index)+1
    src = edge_index[0,:]
    dst = edge_index[1,:]
    src = torch.tensor(src)
    dst = torch.tensor(dst)
    x = torch.tensor(np.random.randn(num_nodes, 1000))
    x = x.to(device)
    src = src.to(device)
    dst = dst.to(device)

    for embedding_dim in embedding:
        print("**********embedding_dim={}**********".format(embedding_dim))
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

        print("**********embedding_dim={}**********".format(embedding_dim))

    print(x.device)