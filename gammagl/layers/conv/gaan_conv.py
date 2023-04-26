import tensorlayerx as tlx
from gammagl.layers.conv import MessagePassing
from gammagl.mpops import *
from gammagl.utils import segment_softmax


class GaANConv(MessagePassing):
    r"""The gated attentional operator from the `"Gated Attention Networks for Learning on Large and Spatiotemporal Graphs"
    <https://arxiv.org/abs/1803.07294>`_ paper

    .. math::
        \mathbf{y}_i = \mathbf{FC}_{\theta_o}(\mathbf{x}_{i}\oplus
        {||}^K_{k=1}\mathbf{g}_i^{(k)}\sum_{j \in \mathcal{N}i} \mathbf{w}_{i,j}^{(k)}
        \mathbf{FC}^{h}_{\theta_v^{(k)}}(\mathbf{z}_{j})),

    where the attention coefficients :math:`\mathbf{w}_{i,j}` are computed as

    .. math::
        \mathbf{w}_{i,j}^{(k)} =
        \frac{
        \exp\left(\phi_w^{(k)}\left(\mathbf{x}_i, \mathbf{z}_j\right)\right)}
        {\sum_{l=1}^{\Vert \mathcal{N}i\Vert}
        \exp\left(\phi_w^{(k)}\left(\mathbf{x}_i, \mathbf{z}_l\right)\right)}.

    where coefficients \phi_w^{(k)}(\mathbf{x},\mathbf{z}) are compute as

    .. math::
        \phi_w^{(k)}(\mathbf{x},\mathbf{z})=
        <\mathbf{FC}_{\theta_{xa}^{(k)}}(\mathbf{x}),
        \mathbf{FC}_{\theta_{za}^{(k)}}(\mathbf{z})>,

    and the gated attention aggregators \mathbf{g}_i are compute as

    .. math::
        \mathbf{g}_i=\mathbf{FC}_{{\theta}_g}^\sigma
        (\mathbf{x}_i\oplus\mathop{max}\limits_{j\in \mathcal{N}_i}
        (\{\mathbf{FC}_{{\theta}_m}\mathbf{z}_j)\})\oplus
        \frac
        {\sum_{j\in \mathcal{N}_i}\mathbf{z}_j}
        {|\mathcal{N}_i|}).

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
        (default: :obj:`2`)
    m: int, optional
        Number of output dimension for gated attention aggregator to then
        take the element wise max.
        (default: :obj:`64`)
    v: int, optional
        Number of output dimension for attention aggregator to record neighbors'message.
        (default: :obj:`64`)
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

    def __init__(self, in_channels, out_channels, heads=8, m=64, v=64,
                 negative_slope=0.1, dropout_rate=0.1, add_bias=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.m = m
        self.v = v
        self.negative_slope = negative_slope
        self.dropout_rate = dropout_rate
        self.add_bias = add_bias
        self.leaky_relu = tlx.layers.LeakyReLU(negative_slope)
        self.dropout = tlx.layers.Dropout(self.dropout_rate)

        self.lin = tlx.layers.Linear(
            in_features=self.in_channels, out_features=self.v * self.heads, act=self.leaky_relu)
        initor = tlx.initializers.TruncatedNormal()

        self.att_src = self._get_weights("att_src", shape=(
            1, self.heads, self.v), init=initor, order=True)
        self.att_dst = self._get_weights("att_dst", shape=(
            1, self.heads, self.v), init=initor, order=True)
        # GaAN layers
        self.g_lin = tlx.layers.Linear(
            in_features=self.in_channels+self.v*self.heads+m, out_features=self.heads, act=tlx.sigmoid)
        self.m_lin = tlx.layers.Linear(
            in_features=self.in_channels, out_features=self.m)
        self.final_lin = tlx.layers.Linear(
            in_features=self.in_channels+self.v*self.heads, out_features=self.heads*self.out_channels)

        if self.add_bias:
            self.bias = self._get_weights("bias", shape=(
                self.heads * self.out_channels,), init=initor)

    def forward(self, x, edge_index, num_nodes=None):
        x = self.propagate(x, edge_index, num_nodes=num_nodes)
        if self.add_bias:
            x += self.bias
        return x

    def message(self, x, edge_index, edge_weight=None, num_nodes=None):
        x_m = self.m_lin(x)
        x_lin = tlx.reshape(self.lin(
            x), shape=(-1, self.heads, self.v))
        # compute attention weight
        node_src = edge_index[0, :]
        node_dst = edge_index[1, :]
        weight_src = tlx.gather(tlx.reduce_sum(
            x_lin * self.att_src, -1), node_src)
        weight_dst = tlx.gather(tlx.reduce_sum(
            x_lin * self.att_dst, -1), node_dst)
        weight = self.leaky_relu(weight_src + weight_dst)

        alpha = self.dropout(segment_softmax(weight, node_dst, num_nodes))
        neighbor = tlx.gather(x_lin, node_src)

        # two term required to compute g_i in aggregate
        g_max = tlx.gather(x_m, node_src)
        g_mean = neighbor

        g_i = (g_max, g_mean)

        attention = neighbor * tlx.expand_dims(alpha, -1)

        return (x, attention * edge_weight, g_i) if edge_weight else (x, attention, g_i)

    def aggregate(self, msg, edge_index, num_nodes=None, aggr='sum'):
        x, attention, g_i = msg
        # x = tlx.reshape(x, shape=(-1, self.out_channels*self.heads))
        dst_index = edge_index[1, :]
        # first aggregation to sum up neighbors' attention value
        att_sum = unsorted_segment_sum(attention, dst_index, num_nodes)
        # second aggregation to assign gate value to each attention head
        g_max, g_mean = g_i
        g_max = unsorted_segment_max(g_max, dst_index, num_nodes)
        g_mean = unsorted_segment_mean(
            g_mean, dst_index, num_nodes)
        g_mean = tlx.reshape(g_mean, shape=(-1, self.v*self.heads))
        g_i = tlx.concat((x, g_max, g_mean), axis=1)

        g_i = tlx.reshape(self.g_lin(g_i), shape=(-1, self.heads, 1))
        # print(g_i.shape)
        # print(att_sum.shape)

        att_sum = tlx.reshape(
            g_i*att_sum, shape=(-1, self.v*self.heads))
        # concat to have final aggregated message
        x = tlx.concat((x, att_sum), axis=1)
        x = self.final_lin(x)
        return x
