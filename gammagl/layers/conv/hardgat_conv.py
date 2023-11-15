import tensorlayerx as tlx
from gammagl.layers.conv import MessagePassing
from gammagl.utils import segment_softmax
import math
import numpy as np


class HardGATConv(MessagePassing):
    r"""The graph hard attentional operator from the `"Graph Representation Learning via Hard and Channel-Wise Attention Networks"
    <https://dl.acm.org/doi/pdf/10.1145/3292500.3330897>`_ paper

    .. math::
        \begin{aligned}
        &y=\frac{\left|X^T p\right|}{\|p\|}\\
        &\text { for } i=1,2, \cdots, N \text { do }\\
        &\quad\quad id x_i=\text { Ranking }_k\left(A_{: i} \circ y\right) \quad \in \mathbb{R}^k\\
        &\quad\quad\hat{X}_i=X\left(:, i d x_i\right) \quad \in \mathbb{R}^{d \times k}\\
        &\quad\quad\tilde{y}_i=\operatorname{sigmoid}\left(y\left(i d x_i\right)\right) \quad \in \mathbb{R}^k\\
        &\quad\quad\tilde{X}_i=\hat{X}_i \operatorname{diag}\left(\tilde{y}_i\right) \quad \in \mathbb{R}^{d \times k}\\
        &\quad\quad z_i=\operatorname{attn}\left(x_i, \tilde{X}_i, \tilde{X}_i\right) \quad \in \mathbb{R}^d\\
        &Z=\left[z_1, z_2, \ldots, z_N\right]\in \mathbb{R}^{d \times N}
        \end{aligned}

    where the attn operation is the same as GAT, and the process is as follows.

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
    in_channels: int, tuple
        Size of each input sample, or :obj:`-1` to
        derive the size from the first input(s) to the forward method.
        A tuple corresponds to the sizes of source and target
        dimensionalities.
    out_channels: int
        Size of each output sample.
    heads: int, optional
        Number of multi-head-attentions.
        (default: :obj:`1`)
    k: int, optional
        Number of neighbors to attention
        (default: :obj:`8`)
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
                 k=8,
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
        self.k = k
        # self.add_self_loops = add_self_loops
        self.add_bias = add_bias

        self.linear = tlx.layers.Linear(out_features=self.out_channels * self.heads,
                                        in_features=self.in_channels,
                                        b_init=None)
        initor = tlx.initializers.XavierNormal(gain=math.sqrt(2.0))
        self.p = self._get_weights("proj", shape=(1, self.in_channels), init=initor, order=True)
        initor = tlx.initializers.TruncatedNormal()
        self.att_src = self._get_weights("att_src", shape=(1, self.heads, self.out_channels), init=initor, order=True)
        self.att_dst = self._get_weights("att_dst", shape=(1, self.heads, self.out_channels), init=initor, order=True)

        self.leaky_relu = tlx.layers.LeakyReLU(negative_slope)
        self.dropout = tlx.layers.Dropout(self.dropout_rate)

        if self.add_bias and concat:
            self.bias = self._get_weights("bias", shape=(self.heads * self.out_channels,), init=initor)
        elif self.add_bias and not concat:
            self.bias = self._get_weights("bias", shape=(self.out_channels,), init=initor)

    def select_topk(self, edge_index, value):
        # Select Top k neighbors, return new edge_index
        src, dst = tlx.convert_to_numpy(edge_index)
        score = tlx.convert_to_numpy(value)
        score = score[src]
        # Sort by value in descending
        rank = np.argsort(score)[::-1]
        src = src[rank]
        dst = dst[rank]
        # Sort by dst in ascending
        index = np.argsort(dst)
        src = src[index]
        dst = dst[index]
        # Each dst-node choose the-top-k edges
        e_id = []
        rowptr = np.concatenate(([0], np.bincount(dst).cumsum()))
        for dst_node in np.unique(dst):
            st = rowptr[dst_node]
            len1 = rowptr[dst_node + 1] - st
            for i in range(st, min(st + self.k, st + len1)):
                e_id.append(i)

        return tlx.convert_to_tensor(np.array([src[e_id], dst[e_id]]))


    def message(self, x, edge_index, edge_weight=None, num_nodes=None):
        node_src = edge_index[0, :]
        node_dst = edge_index[1, :]
        weight_src = tlx.gather(tlx.reduce_sum(x * self.att_src, -1), node_src)
        weight_dst = tlx.gather(tlx.reduce_sum(x * self.att_dst, -1), node_dst)
        weight = self.leaky_relu(weight_src + weight_dst)

        alpha = self.dropout(segment_softmax(weight, node_dst, num_nodes))
        x = tlx.gather(x, node_src) * tlx.expand_dims(alpha, -1)
        return x * edge_weight if edge_weight else x

    def forward(self, x, edge_index, num_nodes):
        # projection process to get importance vector y
        y = tlx.abs(tlx.squeeze(tlx.matmul(self.p, tlx.transpose(x)), axis=0)) / math.sqrt(tlx.ops.reduce_sum(self.p ** 2))
        edge_index = self.select_topk(edge_index, y)
        # Sigmoid as information threshold
        y = tlx.sigmoid(y)
        # Using vector matrix elementwise mul for acceleration
        x = tlx.reshape(y, shape=(-1, 1)) * x
        # attn operation same as 'gat_conv.py'
        x = tlx.reshape(self.linear(x), shape=(-1, self.heads, self.out_channels))
        x = self.propagate(x, edge_index, num_nodes=num_nodes)

        if self.concat:
            x = tlx.reshape(x, (-1, self.heads * self.out_channels))
        else:
            x = tlx.reduce_mean(x, axis=1)

        if self.add_bias:
            x += self.bias
        return x
