import tensorlayerx as tlx
from gammagl.layers.conv import MessagePassing
from gammagl.utils import segment_softmax
from gammagl.mpops import bspmm


class GATConv(MessagePassing):
    r"""The graph attentional operator from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{i} +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j},

    where the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j]
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k]
        \right)\right)}.

    Parameters
    ----------
    in_channels: int or tuple
        Size of each input sample, or :obj:`-1` to
        derive the size from the first input(s) to the forward method.
        A tuple corresponds to the sizes of source and target
        dimensionalities.
    out_channels: int
        Size of each output sample.
    heads: int, optional
        Number of multi-head-attentions.
        (default: :obj:`1`)
    concat: bool, optional
        If set to :obj:`False`, the multi-head
        attentions are averaged instead of concatenated.
        (default: :obj:`True`)
    negative_slope: float, optional
        LeakyReLU angle of the negative
        slope. (default: :obj:`0.2`)
    dropout_rate: float, optional
        Dropout probability of the normalized
        attention coefficients which exposes each node to a stochastically
        sampled neighborhood during training. (default: :obj:`0`)
    add_self_loops: bool, optional
        If set to :obj:`False`, will not add
        self-loops to the input graph. (default: :obj:`True`)
    add_bias: bool, optional
        If set to :obj:`False`, the layer will not learn
        an additive bias. (default: :obj:`True`)

    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 heads=1,
                 concat=True,
                 negative_slope=0.2,
                 dropout_rate=0.,
                 add_bias=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negetive_slop = negative_slope
        self.dropout_rate = dropout_rate
        # self.add_self_loops = add_self_loops
        self.add_bias = add_bias

        self.linear = tlx.layers.Linear(out_features=self.out_channels * self.heads,
                                        in_features=self.in_channels,
                                        b_init=None)
        
        init_weight = tlx.initializers.TruncatedNormal()
        self.w = tlx.nn.Parameter(
            init_weight((in_channels, self.out_channels * self.heads)))

        initor = tlx.initializers.TruncatedNormal()
        self.att = tlx.nn.Parameter(
            initor((1, self.heads, self.out_channels * 2)))

        self.leaky_relu = tlx.layers.LeakyReLU(negative_slope)
        self.dropout = tlx.layers.Dropout(self.dropout_rate)

        if self.add_bias and concat:
            self.bias = self._get_weights("bias", shape=(self.heads * self.out_channels,), init=initor)
        elif self.add_bias and not concat:
            self.bias = self._get_weights("bias", shape=(self.out_channels,), init=initor)

    def forward(self, x, edge_index, num_nodes=None):
        x = tlx.matmul(x, self.w)
        x = tlx.reshape(x, shape=(-1, self.heads, self.out_channels))
        node_src = edge_index[0, :]
        node_dst = edge_index[1, :]
        feat_src = tlx.gather(x, node_src)
        feat_dst = tlx.gather(x, node_dst)
        feat = tlx.concat((feat_src, feat_dst), axis=-1)
        feat = tlx.reshape(feat, shape=(-1, self.heads, self.out_channels * 2))
        e = tlx.reduce_sum(feat * self.att, axis = -1)

        e = self.leaky_relu(e)
        alpha = self.dropout(segment_softmax(e, node_dst, num_nodes))

        x = self.propagate(x, edge_index, num_nodes=num_nodes, edge_weight=alpha)
        # x = bspmm(edge_index, weight=alpha, x=x, reduce='sum')

        if self.concat:
            x = tlx.reshape(x, (-1, self.heads * self.out_channels))
        else:
            x = tlx.reduce_mean(x, axis=1)

        if self.add_bias:
            x += self.bias
        return x

    # def message_aggregate(self, x, edge_index, edge_weight=None, aggr="sum"):
    #     if edge_weight is None:
    #         edge_weight = tlx.ones(shape=(tlx.get_tensor_shape(edge_index)[1],), dtype=tlx.float32)
    #     out = bspmm(edge_index, edge_weight, x, aggr)

    #     return out
