import torch
import dgl.ops as F
import dgl
import numpy as np
import time

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

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

    g = dgl.graph((edge_index[0], edge_index[1]))

    for embedding_dim in embedding:
        print("**********embedding_dim={}**********".format(embedding_dim))
        x = torch.tensor(np.random.randn(num_nodes, embedding_dim))
        x = x.to(device)

        start = time.time()
        for j in range(10):
            F.copy_u_sum(g, x)
        end = time.time()
        print("copy_u_sum:{:.3f}".format(end - start))

        start = time.time()
        for j in range(10):
            F.copy_u_mean(g, x)
        end = time.time()
        print("copy_u_mean:{:.3f}".format(end - start))

        start = time.time()
        for j in range(10):
            F.copy_u_max(g, x)
        end = time.time()
        print("copy_u_max:{:.3f}".format(end - start))

        print("**********embedding_dim={}**********".format(embedding_dim))

    print(x.device)

