import tensorlayerx as tlx

from gammagl.layers.conv import MessagePassing


class EdgeConv(MessagePassing):
    r"""The Edge Convolution operator from the `"Dynamic Graph CNN for Learning on Point Clouds"
    <https://arxiv.org/pdf/1801.07829.pdf>`_ paper

    .. math::
        \mathbf{x}^{(k)}_i = \max_{j\in N(i)}h_\Theta(\mathbf x_i^{(k-1)},x_j^{(k-1)}-x_i^{(k-1)})

    where :math:`\mathbf{x}^{(k)}_i` denotes k-th layer's vector i, and :math:`h_\Theta` denotes a
    multilayer perceptron.

    Parameters
    ----------
    nn (tlx.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
        maps pair-wise concatenated node features :obj:`x` of shape
        :obj:`[-1, 2 * in_channels]` to shape :obj:`[-1, out_channels]`,
        *e.g.*, defined by :class:`tlx.nn.Sequential`.
    aggr (string, optional): The aggregation scheme to use
        (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
        (default: :obj:`"sum"`)
    **kwargs (optional): Additional arguments of
        :class:`gammagl.layers.conv.MessagePassing`.
    """
    def __init__(self, nn, aggr = 'max', **kwargs):
        super().__init__(**kwargs)
        self.aggr = aggr
        self.nn = nn

    def forward(self, x, edge_index):
        """"""
        if not isinstance(x, tuple):
            x = (x, x)
        # propagate_type: (x: PairTensor)
        return self.propagate(x=x, edge_index = edge_index, num_nodes = int(tlx.get_tensor_shape(x[0])[0]))

    def message(self, x):
        x_i = x[0]
        x_j = x[1]
        return self.nn(tlx.concat([x_i, x_j - x_i], axis=-1))

