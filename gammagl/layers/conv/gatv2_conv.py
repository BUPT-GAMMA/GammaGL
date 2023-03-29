import tensorlayerx as tlx
from gammagl.layers.conv import MessagePassing
from gammagl.utils import segment_softmax


class GATV2Conv(MessagePassing):
    r"""The GATv2 operator from the `"How Attentive are Graph Attention Networks?"
    <https://arxiv.org/abs/2105.14491>`_ paper, which fixes the static
    attention problem of the standard :class:`~gammagl.conv.GATConv`
    layer: since the linear layers in the standard GAT are applied right after
    each other, the ranking of attended nodes is unconditioned on the query
    node. In contrast, in GATv2, every node can attend to any other node.

    .. math::
        \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{i} +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j},

    where the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathbf{a}^{\top}\mathrm{LeakyReLU}\left(\mathbf{\Theta}
        [\mathbf{x}_i \, \Vert \, \mathbf{x}_j]
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathbf{a}^{\top}\mathrm{LeakyReLU}\left(\mathbf{\Theta}
        [\mathbf{x}_i \, \Vert \, \mathbf{x}_k]
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

        initor = tlx.initializers.TruncatedNormal()
        self.att_src = self._get_weights("att_src", shape=(1, self.heads, self.out_channels), init=initor,order=True)
        self.att_dst = self._get_weights("att_dst", shape=(1, self.heads, self.out_channels), init=initor,order=True)

        self.leaky_relu = tlx.layers.LeakyReLU(negative_slope)
        self.dropout = tlx.layers.Dropout(self.dropout_rate)

        if self.add_bias and concat:
            self.bias = self._get_weights("bias", shape=(self.heads * self.out_channels,), init=initor)
        elif self.add_bias and not concat:
            self.bias = self._get_weights("bias", shape=(self.out_channels,), init=initor)

    def message(self, x, edge_index, edge_weight=None, num_nodes=None):
        node_src = edge_index[0, :]
        node_dst = edge_index[1, :]
        weight_src = self.leaky_relu(tlx.gather(x, node_src))
        weight_dst = self.leaky_relu(tlx.gather(x, node_dst))
        weight = tlx.reduce_mean(weight_src * self.att_src + weight_dst * self.att_dst, -1)

        alpha = segment_softmax(weight, node_dst, num_nodes)
        alpha = self.dropout(alpha)

        x = tlx.gather(x, node_src) * tlx.expand_dims(alpha, -1)
        return x * edge_weight if edge_weight else x


    def forward(self, x, edge_index, num_nodes=None):
        x = tlx.reshape(self.linear(x), shape=(-1, self.heads, self.out_channels))
        x = self.propagate(x, edge_index, num_nodes=num_nodes)

        if self.concat:
            x = tlx.reshape(x, (-1, self.heads * self.out_channels))
        else:
            x = tlx.reduce_mean(x, axis=1)

        if self.add_bias:
            x += self.bias
        return x

