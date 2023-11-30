import tensorlayerx as tlx
from gammagl.layers.conv import MessagePassing
from gammagl.utils import add_self_loops, calc_gcn_norm


class SAGEConv(MessagePassing):
    r"""The GraphSAGE operator from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i + \mathbf{W}_2 \cdot \
        \mathrm{mean}_{j \in \mathcal{N(i)}} \mathbf{x}_j

    Parameters
    ----------
    in_channels: int, tuple
        Size of each input sample, or :obj:`-1` to
        derive the size from the first input(s) to the forward method.
        A tuple corresponds to the sizes of source and target
        dimensionalities.
    out_channels: int
        Size of each output sample.
    norm: callable, None, optional
        If not None, applies normalization to the updated node features.
    aggr: str, optional
        Aggregator type to use (``mean``).
    add_bias: bool, optional
        If set to :obj:`False`, the layer will not learn
        an additive bias. (default: :obj:`True`)

    """

    def __init__(self, in_channels, out_channels, activation=None, aggr="mean", add_bias=True):
        super(SAGEConv, self).__init__()
        #
        self.aggr = aggr
        self.act = activation
        self.in_feat = in_channels
        # relu use he_normal
        initor = tlx.initializers.he_normal()
        # self and neighbor
        self.fc_neigh = tlx.nn.Linear(in_features=in_channels, out_features=out_channels, W_init=initor, b_init=None)
        if aggr != 'gcn':
            self.fc_self = tlx.nn.Linear(in_features=in_channels, out_features=out_channels, W_init=initor, b_init=None)

        if aggr == "lstm":
            self.lstm = tlx.nn.LSTM(input_size=in_channels, hidden_size=in_channels, batch_first=True)

        if aggr == "pool":
            self.pool = tlx.nn.Linear(in_features=in_channels, out_features=in_channels, W_init=initor, b_init=None)
        self.add_bias = add_bias
        if add_bias:
            init = tlx.initializers.zeros()
            self.bias = self._get_weights("bias", shape=(1, out_channels), init=init)

    def forward(self, feat, edge):
        r"""
                Compute GraphSAGE layer.

                Parameters
                ----------
                feat : Pair of Tensor
                    The pair must contain two tensors of shape
                    :math:`(N_{in}, D_{in_{src}})` and :math:`(N_{out}, D_{in_{dst}})`.
                Returns
                -------
                Tensor
                    The output feature of shape :math:`(N_{dst}, D_{out})`
                    where :math:`N_{dst}` is the number of destination nodes in the input graph,
                    math:`D_{out}` is size of output feature.
        """
        if isinstance(feat, tuple):
            src_feat = feat[0]
            dst_feat = feat[1]
        else:
            src_feat = feat
            dst_feat = feat
        num_nodes = int(dst_feat.shape[0])
        if self.aggr == 'mean':
            src_feat = self.fc_neigh(src_feat)
            out = self.propagate(src_feat, edge, edge_weight=None, num_nodes=num_nodes, aggr='mean')
        elif self.aggr == 'gcn':
            src_feat = self.fc_neigh(src_feat)
            edge, _ = add_self_loops(edge)
            weight = calc_gcn_norm(edge, int(1 + tlx.reduce_max(edge[0])))
            # col, row, weight = calc(edge,  1 + tlx.reduce_max(edge[0]))
            out = self.propagate(src_feat, edge, edge_weight=weight, num_nodes=int(1 + tlx.reduce_max(edge[0])), aggr='sum')
            out = tlx.gather(out, tlx.arange(0, num_nodes))
        elif self.aggr == 'pool':
            src_feat = tlx.nn.ReLU()(self.pool(src_feat))
            out = self.propagate(src_feat, edge, edge_weight=None, num_nodes=num_nodes, aggr='max')
            out = self.fc_neigh(out)
        elif self.aggr == 'lstm':
            src_feat = tlx.reshape(src_feat, (dst_feat.shape[0], -1, src_feat.shape[1]))
            size = dst_feat.shape[0]
            h = (tlx.zeros((1, size, self.in_feat)),
                 tlx.zeros((1, size, self.in_feat)))
            _, (rst, _) = self.lstm(src_feat, h)
            out = self.fc_neigh(rst[0])
        if self.aggr != 'gcn':
            out += self.fc_self(dst_feat)
        if self.add_bias:
            out += self.bias

        if self.act is not None:
            out = self.act(out)

        return out


