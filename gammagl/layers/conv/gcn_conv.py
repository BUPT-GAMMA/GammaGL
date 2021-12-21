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


class GCNConv(MessagePassing):
    r"""The graph convolutional operator from the `"Semi-supervised
    Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \mathbf{X} \mathbf{\Theta},

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.
    The adjacency matrix can include other values than :obj:`1` representing
    edge weights via the optional :obj:`edge_weight` tensor.

    Its node-wise formulation is given by:

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta} \sum_{j \in \mathcal{N}(v) \cup
        \{ i \}} \frac{e_{j,i}}{\sqrt{\hat{d}_j \hat{d}_i}} \mathbf{x}_j

    with :math:`\hat{d}_i = 1 + \sum_{j \in \mathcal{N}(i)} e_{j,i}`, where
    :math:`e_{j,i}` denotes the edge weight from source node :obj:`j` to target
    node :obj:`i` (default: :obj:`1.0`)

    Parameters:
        in_channels: Size of each input sample
        out_channels: Size of each output sample.
        improved: If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`. (default: :obj:`False`)
        add_self_loops: If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        renorm: Whether to add self-loops and compute
            symmetric normalization coefficients on the fly.
            (default: :obj:`True`)
        add_bias: If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        aggr: way of aggregation(sum, mean, max, min), default=`sum`.
        flow: direction of message passing('source_to_target', 'source_to_target'),
            default=`'source_to_target'`
        node_dim: default=`-2`.
    """

    def __init__(self, 
                in_channels,
                out_channels,
                improved=False,
                add_self_loops=True,
                renorm=True,
                add_bias=True):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.add_self_loops = add_self_loops
        self.renorm = renorm

        self.linear = tl.layers.Dense(n_units=self.out_channels,
                                      in_channels=self.in_channels,
                                      b_init=None)
        if add_bias is True:
            initor = tl.initializers.Zeros()
            self.bias = self._get_weights("bias", shape=(1,self.out_channels), init=initor)

    def message_aggregate(self, x, sparse_adj):
        return sparse_adj @ x

    def forward(self, x, edge_index, edge_weight=None):
        num_nodes = tf.shape(x)[0]
        sparse_adj = SparseAdj(edge_index, edge_weight, [num_nodes, num_nodes])
        sparse_adj = gcn_norm(sparse_adj)

        x = self.linear(x)
        out = self.propagate(x, sparse_adj)
        if self.bias is not None:
            out += self.bias
        
        return out
    
