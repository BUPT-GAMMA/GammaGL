import math
from abc import ABC

import tensorflow as tf
import tensorlayer as tl
from gammagraphlibrary.layers.conv import MessagePassing
from gammagraphlibrary.sparse.sparse_adj import SparseAdj
from gammagraphlibrary.sparse.sparse_ops import sparse_diag_matmul, diag_sparse_matmul

def segment_softmax(data, segment_ids, num_segments):
    max_values = tf.math.unsorted_segment_max(data, segment_ids, num_segments=num_segments)
    gathered_max_values = tf.gather(max_values, segment_ids)
    exp = tf.exp(data - tf.stop_gradient(gathered_max_values))
    denominator = tf.math.unsorted_segment_sum(exp, segment_ids, num_segments=num_segments) + 1e-8
    gathered_denominator = tf.gather(denominator, segment_ids)
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
                 add_bias=True, 
                 node_dim=-2):
        super().__init__(node_dim=node_dim)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negetive_slop = negative_slope
        self.dropout_rate = dropout_rate
        # self.add_self_loops = add_self_loops
        self.add_bias = add_bias

        self.linear_w = tl.layers.Dense(n_units=self.out_channels * self.heads,
                                        in_channels=self.in_channels,
                                        b_init=None)

        initor = tl.initializers.TruncatedNormal()
        self.att_src = self._get_weights("att_src", shape=(1, self.heads, self.out_channels), init=initor)
        self.att_dst = self._get_weights("att_dst", shape=(1, self.heads, self.out_channels), init=initor)

        self.leaky_relu = tl.layers.LeakyReLU(alpha=negative_slope)
        self.dropout = tl.layers.Dropout(keep=1 - self.dropout_rate)

        if self.add_bias and concat:
            self.bias = self._get_weights("bias", shape=(self.heads * self.out_channels,), init=initor)
        elif self.add_bias and not concat:
            self.bias = self._get_weights("bias", shape=(self.out_channels,), init=initor)
        
    def message(self, x, edge_index, num_nodes):
        node_src = tf.concat([edge_index[0, :], tf.range(num_nodes)], axis=0)
        node_dst = tf.concat([edge_index[1, :], tf.range(num_nodes)], axis=0)
        weight_src = tf.gather(tf.reduce_sum(x * self.att_src, -1), node_src)
        weight_dst = tf.gather(tf.reduce_sum(x * self.att_dst, -1), node_dst)
        weight = self.leaky_relu(weight_src + weight_dst)

        # weight = tf.gather(weight, node_src)
        alpha = segment_softmax(weight, node_dst, num_nodes)

        # weight = tf.exp(weight)
        # weight = tf.gather(weight, node_dst)
        # weight_sum = tf.math.unsorted_segment_sum(weight, node_src, num_segments=num_nodes)
        # weight_sum = tf.gather(weight_sum, node_dst)
        # alpha = weight / weight_sum

        alpha = self.dropout(alpha)

        return tf.gather(x, node_src) * tf.expand_dims(alpha, -1)


    def aggregate(self, x, edge_index, num_nodes):
        node_dst = tf.concat([edge_index[1, :], tf.range(num_nodes)], axis=0)
        return tf.math.unsorted_segment_sum(x, node_dst, num_segments=num_nodes)


    def propagate(self, x, edge_index, num_nodes):
        x = self.message(x, edge_index, num_nodes)
        x = self.aggregate(x, edge_index, num_nodes)
        x = self.update(x)
        return x

    def forward(self, x, edge_index, num_nodes):
        x = tf.reshape(self.linear_w(x), shape=(-1, self.heads, self.out_channels))
        x = self.propagate(x, edge_index, num_nodes)

        if self.concat:
            x = tf.reshape(x, (-1, self.heads * self.out_channels))
        else:
            x = tf.reduce_mean(x, axis=1)

        if self.bias is not None:
            x += self.bias
        return x

