import os

os.environ['TL_BACKEND'] = 'torch'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import tensorlayerx as tlx
from gammagl.mpops import *
import time

relative_path = '/home/zgy/review/zgy/GammaGL/profiler/mpops/edge_index/'
file_name = ['cora.npy', 'pubmed.npy', 'ogbn-arxiv.npy']
embedding = [16, 64, 256]
heads = [8, 16, 32, 64]
iter = 100


with open('test_results.txt', 'w') as result_file:
    for name in file_name:
        path = relative_path + name
        info = f"Loading data from {path}\n"
        result_file.write(info)
        print(info)

        edge_index = np.load(path)

        num_nodes = np.max(edge_index) + 1
        src = tlx.convert_to_tensor(edge_index[0, :], tlx.int64)
        dst = tlx.convert_to_tensor(edge_index[1, :], tlx.int64)
        edge_index = tlx.convert_to_tensor(edge_index)

        for head in heads:

            weight = torch.ones((edge_index.shape[1], head), dtype=tlx.float32)

            for embedding_dim in embedding:
                info = f"**********embedding_dim={embedding_dim}  head={head}**********\n"
                result_file.write(info)
                print(info)
                x = tlx.convert_to_tensor(np.random.randn(num_nodes, head, embedding_dim), dtype=tlx.float32)

                start = time.time()
                for j in range(iter):
                    bspmm(edge_index, weight=weight, x=x, reduce='sum')
                end = time.time()
                info = "bspmm_sum:{:.3f}\n".format(end-start)
                result_file.write(info)
                print(info)

                start = time.time()
                for j in range(iter):
                    msg = tlx.gather(x, src)
                    edge_weight = tlx.expand_dims(weight, -1)
                    msg = msg * edge_weight
                    unsorted_segment_sum(msg, dst, num_nodes)
                end = time.time()
                info = "segment_sum:{:.3f}\n".format(end-start)
                result_file.write(info)
                print(info)


                info = f"**********embedding_dim={embedding_dim}  head={head}**********\n"
                result_file.write(info)
                print(info)

        info = f"Data tensors are on device: {x.device}\n"
        result_file.write(info)
        print(info)
