import tensorflow as tf
import tensorlayer as tl
from gammagl.layers.conv import MessagePassing
from gammagl.sparse.sparse_adj import SparseAdj
from gammagl.sparse.sparse_ops import sparse_diag_matmul, diag_sparse_matmul


class SGConv(MessagePassing):
    """simple graph convolution"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 itera_K=2):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.itera_K = itera_K

        self.linear = tl.layers.Dense(n_units=self.out_channels,
                                      in_channels=self.in_channels,
                                      b_init=None)

    def message_aggregate(self, x, sparse_adj):
        return sparse_adj @ x

    def forward(self, x, sparse_adj, edge_weight=None):
        x = self.linear(x)
        for _ in range(self.itera_K):
            x = sparse_adj @ x

        return x
