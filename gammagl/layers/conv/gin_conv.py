import tensorlayerx as tlx
from gammagl.layers.conv import MessagePassing


class GINConv(MessagePassing):
    r"""The graph isomorphism operator from the `"How Powerful are
    Graph Neural Networks?" <https://arxiv.org/abs/1810.00826>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = h_{\mathbf{\Theta}} \left( (1 + \epsilon) \cdot
        \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \right)

    or

    .. math::
        \mathbf{X}^{\prime} = h_{\mathbf{\Theta}} \left( \left( \mathbf{A} +
        (1 + \epsilon) \cdot \mathbf{I} \right) \cdot \mathbf{X} \right),

    here :math:`h_{\mathbf{\Theta}}` denotes a neural network, *.i.e.* an MLP.

    Parameters
    ----------
    nn: tlx.nn.Module
        A neural network :math:`h_{\mathbf{\Theta}}` that
        maps node features :obj:`x` of shape :obj:`[-1, in_channels]` to
        shape :obj:`[-1, out_channels]`, *e.g.*, defined by
        :class:`torch.nn.Sequential`.
    eps: float, optional
        (Initial) :math:`\epsilon`-value.
        (default: :obj:`0.`)
    train_eps: bool, optional
        If set to :obj:`True`, :math:`\epsilon`
        will be a trainable parameter. (default: :obj:`False`)
    **kwargs: optional
        Additional arguments of
        :class:`gammagl.layers.conv.MessagePassing`.

    """

    def __init__(self, nn, eps=0., train_eps: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.nn = nn
        if train_eps:
            self.eps = tlx.nn.Parameter(tlx.convert_to_tensor([eps]))
        else:
            self.eps = tlx.convert_to_tensor([eps])

    def forward(self, x, edge_index, size=None):
        if not isinstance(x, (list, tuple)):
            x = (x, x)
        assert len(x) == 2

        out = self.propagate(x=x[0], edge_index=edge_index, size=size)

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r

        return self.nn(out)

    def message(self, x, edge_index):
        return tlx.gather(x, edge_index[0, :])
