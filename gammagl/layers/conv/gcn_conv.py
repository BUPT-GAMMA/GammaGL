import tensorlayerx as tlx
from gammagl.utils import degree
from gammagl.layers.conv import MessagePassing



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

    Parameters
    ----------
    in_channels: int
        Size of each input sample.
    out_channels: int
        Size of each output sample.
    norm: str, optional
        How to apply the normalizer.  Can be one of the following values:
        
            * ``right``, to divide the aggregated messages by each node's in-degrees, which is equivalent to averaging the received messages.

            * ``none``, where no normalization is applied.

            * ``both`` (default), where the messages are scaled with :math:`1/c_{ji}` above, equivalent to symmetric normalization.

            * ``left``, to divide the messages sent out from each node by its out-degrees, equivalent to random walk normalization.
    add_bias: bool, optional
        If set to :obj:`False`, the layer will not learn
        an additive bias. (default: :obj:`True`)

    """

    def __init__(self, 
                in_channels,
                out_channels,
                norm = 'both',
                add_bias = True):
        super().__init__()
        
        if norm not in ['left', 'right', 'none', 'both']:
            raise ValueError('Invalid norm value. Must be either "none", "both", "right" or "left".'
                             ' But got "{}".'.format(norm))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.add_bias = add_bias
        self._norm = norm

        self.linear = tlx.layers.Linear(out_features=out_channels,
                                        in_features=in_channels,
                                        W_init='xavier_uniform',
                                        b_init=None)
        if add_bias is True:
            initor = tlx.initializers.Zeros()
            self.bias = self._get_weights("bias", shape=(1,self.out_channels), init=initor)

    def forward(self, x, edge_index, edge_weight=None, num_nodes=None):
        x = self.linear(x)
        src, dst = edge_index[0], edge_index[1]
        if edge_weight is None:
            edge_weight = tlx.ones(shape=(edge_index.shape[1], 1))
        edge_weight = tlx.reshape(edge_weight,(-1,))
        weights = edge_weight
        num_nodes = tlx.reduce_max(edge_index) + 1
        
        if self._norm in ['left', 'both']:
            deg = degree(src, num_nodes=num_nodes, dtype = tlx.float32)
            if self._norm == 'both':
                norm = tlx.pow(deg, -0.5)
            else:
                norm = 1.0 / deg
            weights = tlx.ops.gather(norm, src) * tlx.reshape(edge_weight, (-1,))

        if self._norm in ['right', 'both']:
            deg = degree(dst, num_nodes=num_nodes, dtype = tlx.float32)
            if self._norm == 'both':
                norm = tlx.pow(deg, -0.5)
            else:
                norm = 1.0 / deg
            weights = tlx.reshape(weights, (-1,)) * tlx.ops.gather(norm, dst)

        out = self.propagate(x, edge_index, edge_weight=weights, num_nodes=num_nodes)
        if self.add_bias:
            out += self.bias
        
        return out

