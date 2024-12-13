import tensorlayerx as tlx
from gammagl.layers.conv import MessagePassing


class SGConv(MessagePassing):
    r"""The simple graph convolutional operator from the `"Simplifying Graph
    Convolutional Networks" <https://arxiv.org/abs/1902.07153>`_ paper

    .. math::
        \mathbf{X}^{\prime} = {\left(\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \right)}^K \mathbf{X} \mathbf{\Theta},

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.
    The adjacency matrix can include other values than :obj:`1` representing
    edge weights via the optional :obj:`edge_weight` tensor.

    Parameters
    ----------
    in_channels: int
        Size of each input sample, or :obj:`-1` to derive
        the size from the first input(s) to the forward method.
    out_channels: int
        Size of each output sample.
    iter_K: int, optional
        Number of hops :math:`K`. (default: :obj:`1`)

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 iter_K=2):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.iter_K = iter_K

        self.linear = tlx.layers.Linear(out_features=self.out_channels,
                                      in_features=self.in_channels,
                                      b_init=None)

    def forward(self, x, edge_index, edge_weight=None, num_nodes=None):
        x = self.linear(x)
        for _ in range(self.iter_K):
            x = self.propagate(x, edge_index, edge_weight=edge_weight, num_nodes=num_nodes)

        return x
