import tensorlayerx as tlx
from gammagl.layers.conv import MessagePassing
from .gat_conv import GATConv

def segment_softmax(data, segment_ids, num_segments):
    max_values = tlx.ops.unsorted_segment_max(data, segment_ids, num_segments=num_segments) # tensorlayerx not supported
    gathered_max_values = tlx.ops.gather(max_values, segment_ids)
    # exp = tlx.ops.exp(data - tf.stop_gradient(gathered_max_values))
    exp = tlx.ops.exp(data - gathered_max_values)
    denominator = tlx.ops.unsorted_segment_sum(exp, segment_ids, num_segments=num_segments) + 1e-8
    gathered_denominator = tlx.ops.gather(denominator, segment_ids)
    score = exp / (gathered_denominator + 1e-16)
    return score

class SuperGATConv(MessagePassing):
    """
    The self-supervised graph attentional operator from the `"How to Find
    Your Friendly Neighborhood: Graph Attention Design with Self-Supervision"
    <https://openreview.net/forum?id=Wi5KUNlqWty>`_ paper

    .. math::

        \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{i} +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j},

    where the two types of attention :math:`\alpha_{i,j}^{\mathrm{MX\ or\ SD}}`
    are computed as:

    .. math::

        \alpha_{i,j}^{\mathrm{MX\ or\ SD}} &=
        \frac{
        \exp\left(\mathrm{LeakyReLU}\left(
            e_{i,j}^{\mathrm{MX\ or\ SD}}
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathrm{LeakyReLU}\left(
            e_{i,k}^{\mathrm{MX\ or\ SD}}
        \right)\right)}

        e_{i,j}^{\mathrm{MX}} &= \mathbf{a}^{\top}
            [\mathbf{\Theta}\mathbf{x}_i \, \Vert \,
             \mathbf{\Theta}\mathbf{x}_j]
            \cdot \sigma \left(
                \left( \mathbf{\Theta}\mathbf{x}_i \right)^{\top}
                \mathbf{\Theta}\mathbf{x}_j
            \right)

        e_{i,j}^{\mathrm{SD}} &= \frac{
            \left( \mathbf{\Theta}\mathbf{x}_i \right)^{\top}
            \mathbf{\Theta}\mathbf{x}_j
        }{ \sqrt{d} }

    The self-supervised task is a link prediction using the attention values
    as input to predict the likelihood :math:`\phi_{i,j}^{\mathrm{MX\ or\ SD}}`
    that an edge exists between nodes:

    .. math::

        \phi_{i,j}^{\mathrm{MX}} &= \sigma \left(
            \left( \mathbf{\Theta}\mathbf{x}_i \right)^{\top}
            \mathbf{\Theta}\mathbf{x}_j
        \right)

        \phi_{i,j}^{\mathrm{SD}} &= \sigma \left(
            \frac{
                \left( \mathbf{\Theta}\mathbf{x}_i \right)^{\top}
                \mathbf{\Theta}\mathbf{x}_j
            }{ \sqrt{d} }
        \right)

    .. note::

        For an example of using SuperGAT, see `examples/super_gat.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        super_gat.py>`_.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 heads=1,
                 concat=True,
                 negative_slope=0.2,
                 dropout_rate=0,
                 attention_type='MX',
                 neg_sample_ratio=0.5,
                 edge_sample_ratio=1,
                 add_bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negetive_slop = negative_slope
        self.dropout_rate = dropout_rate
        self.attention_type = attention_type
        self.neg_sample_ratio = neg_sample_ratio
        self.edge_sampel_ratio = edge_sample_ratio
        self.add_bias = add_bias

        self.linear_w = tlx.layers.Dense(n_units=self.out_channels * self.heads,
                                         in_channels=self.in_channels,
                                         b_init=None)
        self.leaky_relu = tlx.layers.LeakyReLU(alpha=negative_slope)
        initor = tlx.initializers.TruncatedNormal()
        if self.attention_type == 'MX':
            self.att_src = self._get_weights("att_src", shape=(1, self.heads, self.out_channels), init=initor)
            self.att_dst = self._get_weights("att_dst", shape=(1, self.heads, self.out_channels), init=initor)

        if self.add_bias and concat:
            self.bias = self._get_weights("bias", shape=(self.heads * self.out_channels,), init=initor)
        elif self.add_bias and not concat:
            self.bias = self._get_weights("bias", shape=(self.out_channels,), init=initor)

    def message(self, x, edge_index, edge_weight=None, num_nodes=None):
        node_src = edge_index[0, :]
        node_dst = edge_index[1, :]
        x_src = tlx.ops.gather(x, node_src)
        x_dst = tlx.ops.gather(x, node_dst)
        alpha = self.calc_attention(x_src, x_dst, node_dst, num_nodes=num_nodes)


    def forward(self, x, edge_index,neg_edge_index=None, num_nodes=None, batch=None):
        x = tlx.ops.reshape(self.linear_w(x), shape=(-1, self.heads, self.out_channels))
        x = self.propagate(x, edge_index, num_nodes=num_nodes)

        if self.is_train:
            pos_edge_index = self.pos_sampling(edge_index)
            pos_att = self.calc_attention(
                x_src=tlx.ops.gather(x, pos_edge_index[0,:]),
                x_dst=tlx.ops.gather(x, pos_edge_index[1,:]),
                node_dst=pos_edge_index[1,:],
                return_logits=True,
                num_nodes=num_nodes
            )

            if neg_edge_index is None:
                neg_edge_index = self.negative_sampling(edge_index, num_nodes, batch)
            neg_att = self.calc_attention(
                x_src=tlx.ops.gather(x, neg_edge_index[0, :]),
                x_dst=tlx.ops.gather(x, neg_edge_index[1, :]),
                node_dst=neg_edge_index[1, :],
                return_logits=True,
                num_nodes=num_nodes
            )
            self.x_att = tlx.ops.concat([pos_att, neg_att], axis=0)
            self.y_att = self.x_att.new_zeros(self.x_att.shape[0])
            import numpy as np
            np.zeros(self.x_att.shape[0])
            self.y_att[:pos_edge_index.shape[1]] = 1.

        if self.concat:
            x = tlx.ops.reshape(x, (-1, self.heads * self.out_channels))
        else:
            x = tlx.ops.reduce_mean(x, axis=1)

        if self.bias is not None:
            x += self.bias
        return x

    def calc_attention(self, x_src, x_dst, node_dst, return_logits=False, num_nodes=None):
        if self.attention_type == 'MX':
            logits = tlx.reduce_sum(x_src*x_dst, axis=-1)
            if return_logits:
                return logits

            alpha = tlx.reduce_sum(x_src*self.att_src, axis=-1) + tlx.reduce_sum(x_dst*self.att_dst, axis=-1)
            alpha = alpha * tlx.ops.sigmoid(logits)
        else:
            alpha = tlx.reduce_sum(x_src * x_dst, axis=-1) / tlx.ops.sqrt(self.out_channels)
            if return_logits:
                return alpha

        alpha = self.leaky_relu(alpha)
        alpha = segment_softmax(alpha, node_dst, num_segments=num_nodes)
        return alpha

    def calc_attention_loss(self):
        if not self.is_train:
            return tlx.convert_to_tensor([0])

        return tlx.losses.softmax_cross_entropy_with_logits(
            tlx.reduce_mean(self.x_sample, axis=-1),
            self.y_sample,
        )

    def pos_sampling(self, edge_index):
        return

    def neg_sampling(self, edge_index, num_nodes, batch):
        return


