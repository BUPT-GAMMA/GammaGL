import tensorlayer as tl
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

    Parameters:
        in_channels: Size of each input sample
        out_channels: Size of each output sample.
        add_bias: If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(self, 
                in_channels,
                out_channels,
                add_bias=True):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.linear = tl.layers.Dense(n_units=self.out_channels,
                                      in_channels=self.in_channels,
                                      b_init=None)
        if add_bias is True:
            initor = tl.initializers.Zeros()
            self.bias = self._get_weights("bias", shape=(1,self.out_channels), init=initor)

    def message_aggregate(self, x, sparse_adj):
        return sparse_adj @ x

    def forward(self, x, sparse_adj, edge_weight=None):
        x = self.linear(x)
        out = self.propagate(x, sparse_adj)
        if self.bias is not None:
            out += self.bias
        
        return out
    
