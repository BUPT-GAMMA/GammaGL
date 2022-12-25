from tensorlayerx.dataflow import DataLoader
import tensorlayerx as tlx
from gammagl.sample import random_walk,saint_sample
from scipy.sparse import csr_matrix
import numpy as np

class GraphSAINTSampler(DataLoader):
    def __init__(self,data,batch_size,num_steps,sample_coverage) -> None:
        self.num_steps=num_steps
        self.batch_size=batch_size
        self.sample_coverage=sample_coverage
        self.N=N=data.num_nodes
        self.E=data.num_edges
        self.data=data
        self.adj=csr_matrix((tlx.arange(self.E),(data.edge_index[0],data.edge_index[1])),shape=(N,N))
        
#         csr_matrix((data, (row_ind, col_ind)), [shape=(M, N)])
#  |          where ``data``, ``row_ind`` and ``col_ind`` satisfy the
#  |          relationship ``a[row_ind[k], col_ind[k]] = data[k]``.