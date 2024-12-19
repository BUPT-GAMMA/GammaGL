import tensorlayerx as tlx
from tensorlayerx.nn import Linear
from gammagl.layers.conv.message_passing import MessagePassing


class FILMConv(MessagePassing):
    r"""The FiLM graph convolutional operator from the
    `"GNN-FiLM: Graph Neural Networks with Feature-wise Linear Modulation"
    <https://arxiv.org/abs/1906.12192>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \sum_{r \in \mathcal{R}}
        \sum_{j \in \mathcal{N}(i)} \sigma \left(
        \boldsymbol{\gamma}_{r,i} \odot \mathbf{W}_r \mathbf{x}_j +
        \boldsymbol{\beta}_{r,i} \right)

    where :math:`\boldsymbol{\beta}_{r,i}, \boldsymbol{\gamma}_{r,i} =
    g(\mathbf{x}_i)` with :math:`g` being a single linear layer by default.
    Self-loops are automatically added to the input graph and represented as
    its own relation type.

    Parameters
    ----------
    in_channels: int, tuple
        Size of each input sample, or :obj:`-1` to
        derive the size from the first input(s) to the forward method.
        A tuple corresponds to the sizes of source and target
        dimensionalities.
    out_channels: int
        Size of each output sample.
    num_relations: int, optional
        Number of relations. (default: :obj:`1`)
    act: callable, optional
        Activation function :math:`\sigma`.
        (default: :meth:`tlx.nn.ReLU()`)

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_relations=1,
                 act=tlx.nn.ReLU()):
        super(FILMConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.act = act

        self.lins = tlx.nn.ModuleList()
        self.films = tlx.nn.ModuleList()

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
        initor = tlx.initializers.TruncatedNormal()
        for _ in range(num_relations):
            self.lins.append(Linear(in_features=in_channels[0], out_features=out_channels, W_init=initor, b_init=None))
            self.films.append(Linear(in_features=in_channels[1], out_features=2 * out_channels, W_init=initor))

        self.lin_skip = Linear(in_features=in_channels[1], out_features=out_channels, W_init=initor)
        self.film_skip = Linear(in_features=in_channels[1], out_features=2 * out_channels, W_init=initor, b_init=None)

    def forward(self, x, edge_index):

        x = (x, x)

        beta, gamma = tlx.split(self.film_skip(x[1]), 2, axis=-1)
        out = tlx.multiply(gamma, self.lin_skip(x[1])) + beta

        if self.act is not None:
            out = self.act(out)

        beta, gamma = tlx.split(self.films[0](x[1]), 2, axis=-1)
        out = out + self.propagate(x=self.lins[0](x[0]), edge_index=edge_index, beta=beta, gamma=gamma)

        return out

    def message(self, x, edge_index, beta, gamma, edge_weight=None):

        msg = tlx.gather(x, edge_index[1, :])
        beta = tlx.gather(beta, edge_index[0, :])
        gamma = tlx.gather(gamma, edge_index[0, :])

        msg = tlx.multiply(gamma, msg) + beta

        if self.act is not None:
            msg = self.act(msg)
        return msg
