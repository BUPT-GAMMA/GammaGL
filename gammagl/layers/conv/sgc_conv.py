import tensorflow as tf
import tensorlayer as tl
from gammagl.layers.conv import MessagePassing
from gammagl.sparse.sparse_adj import SparseAdj
from gammagl.sparse.sparse_ops import sparse_diag_matmul, diag_sparse_matmul


def gcn_norm(sparse_adj):
    """
    Compute normed edge (updated edge_index and normalized edge_weight) for GCN normalization.

    Parameters:
        sparse_adj: SparseAdj, sparse adjacency matrix.
    
    Returns:
        Normed edge (updated edge_index and normalized edge_weight).
    """

    deg = sparse_adj.reduce_sum(axis=-1)
    deg_inv_sqrt = tl.pow(deg, -0.5)
    deg_inv_sqrt = tl.where(
        tl.logical_or(tl.is_inf(deg_inv_sqrt), tl.is_nan(deg_inv_sqrt)),
        tl.zeros_like(deg_inv_sqrt),
        deg_inv_sqrt
    )

    # (D^(-1/2)A)D^(-1/2)
    normed_sparse_adj = sparse_diag_matmul(diag_sparse_matmul(deg_inv_sqrt, sparse_adj), deg_inv_sqrt)

    return normed_sparse_adj


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

    def forward(self, x, edge_index, edge_weight=None):
        num_nodes = tf.shape(x)[0]
        sparse_adj = SparseAdj(edge_index, edge_weight, [num_nodes, num_nodes])
        sparse_adj = gcn_norm(sparse_adj)

        x = self.linear(x)
        for _ in range(self.itera_K):
            x = sparse_adj @ x

        return x
