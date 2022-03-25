import tensorlayerx as tlx
from gammagl.layers.conv import MessagePassing


def segment_softmax(data, segment_ids, num_segments):
    max_values = tlx.ops.unsorted_segment_max(data, segment_ids, num_segments=num_segments) # tensorlayerx not supported
    gathered_max_values = tlx.ops.gather(max_values, segment_ids)
    # exp = tlx.ops.exp(data - tf.stop_gradient(gathered_max_values))
    exp = tlx.ops.exp(data - gathered_max_values)
    denominator = tlx.ops.unsorted_segment_sum(exp, segment_ids, num_segments=num_segments) + 1e-8
    gathered_denominator = tlx.ops.gather(denominator, segment_ids)
    score = exp / (gathered_denominator + 1e-16)
    return score


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
    
    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout_rate (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        add_bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 heads=1,
                 concat=True,
                 negative_slope=0.2,
                 dropout_rate=0,
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

        self.linear_w = tlx.layers.Dense(n_units=self.out_channels * self.heads,
                                        in_channels=self.in_channels,
                                        b_init=None)

        initor = tlx.initializers.TruncatedNormal()
        self.att_src = self._get_weights("att_src", shape=(1, self.heads, self.out_channels), init=initor)
        self.att_dst = self._get_weights("att_dst", shape=(1, self.heads, self.out_channels), init=initor)

        self.leaky_relu = tlx.layers.LeakyReLU(alpha=negative_slope)
        self.dropout = tlx.layers.Dropout(keep=1 - self.dropout_rate)

        if self.add_bias and concat:
            self.bias = self._get_weights("bias", shape=(self.heads * self.out_channels,), init=initor)
        elif self.add_bias and not concat:
            self.bias = self._get_weights("bias", shape=(self.out_channels,), init=initor)
        
    def message(self, x, edge_index, edge_weight=None, num_nodes=None):
        node_src = edge_index[0, :] # tlx.ops.concat([edge_index[0, :], tlx.ops.range(num_nodes)], axis=0)
        node_dst = edge_index[1, :] # tlx.ops.concat([edge_index[1, :], tlx.ops.range(num_nodes)], axis=0)
        weight_src = tlx.ops.gather(tlx.ops.reduce_sum(x * self.att_src, -1), node_src)
        weight_dst = tlx.ops.gather(tlx.ops.reduce_sum(x * self.att_dst, -1), node_dst)
        weight = self.leaky_relu(weight_src + weight_dst)

        alpha = self.dropout(segment_softmax(weight, node_dst, num_nodes))
        x = tlx.ops.gather(x, node_src) * tlx.ops.expand_dims(alpha, -1)
        return x * edge_weight if edge_weight else x


    def forward(self, x, edge_index, num_nodes):
        x = tlx.ops.reshape(self.linear_w(x), shape=(-1, self.heads, self.out_channels))
        x = self.propagate(x, edge_index, num_nodes=num_nodes)

        if self.concat:
            x = tlx.ops.reshape(x, (-1, self.heads * self.out_channels))
        else:
            x = tlx.ops.reduce_mean(x, axis=1)

        if self.bias is not None:
            x += self.bias
        return x

