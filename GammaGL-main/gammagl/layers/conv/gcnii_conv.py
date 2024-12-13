import tensorlayerx as tlx
from gammagl.layers.conv import MessagePassing

class GCNIIConv(MessagePassing):
    r"""
    The graph convolutional operator with initial residual connections and
    identity mapping (GCNII) from the `"Simple and Deep Graph Convolutional
    Networks" <https://arxiv.org/abs/2007.02133>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \left( (1 - \alpha) \mathbf{\hat{P}}\mathbf{X} +
        \alpha \mathbf{X^{(0)}}\right) \left( (1 - \beta) \mathbf{I} + \beta
        \mathbf{\Theta} \right)

    with :math:`\mathbf{\hat{P}} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
    \mathbf{\hat{D}}^{-1/2}`, where
    :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the adjacency
    matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix,
    and :math:`\mathbf{X}^{(0)}` being the initial feature representation.
    Here, :math:`\alpha` models the strength of the initial residual
    connection, while :math:`\beta` models the strength of the identity
    mapping.
    The adjacency matrix can include other values than :obj:`1` representing
    edge weights via the optional :obj:`edge_weight` tensor.

    Parameters
    ----------
    in_channels: int
        Size of each input sample.
    out_channels: int
        Size of each outoput sample.
    alpha: float
        The strength of the initial residual connection
        :math:`\alpha`.
    beta: float
        The hyperparameter :math:`\beta` to compute \
        the strength of the identity mapping 
        :math:`\beta = \log \left( \frac{\beta}{\ell} + 1 \right)`.
        (default: :obj:`None`)
    variant: bool, optional
        use GCNII*, which can be fomuliazed as following:
 
        .. math::
           \mathbf{H}^{(\ell+1)}= \sigma\left(\left(1-\alpha_{\ell}\right) \tilde{\mathbf{P}} \mathbf{H}^{(\ell)}\left(\left(1-\beta_{\ell}\right) \mathbf{I}_{n}+\beta_{\ell} \mathbf{W}_{1}^{(\ell)}\right)+\right.\\
           \left.+\alpha_{\ell} \mathbf{H}^{(0)}\left(\left(1-\beta_{\ell}\right) \mathbf{I}_{n}+\beta_{\ell} \mathbf{W}_{2}^{(\ell)}\right)\right)

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 alpha,
                 beta,
                 variant=False):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.alpha = alpha
        self.beta = beta
        self.variant = variant
        self.linear = tlx.layers.Linear(out_features=self.out_channels,
                                        in_features=self.in_channels,
                                        b_init=None)
        if self.variant:
            # what if same as linear
            self.linear0 = tlx.layers.Linear(out_features=self.out_channels,
                                             in_features=self.in_channels,
                                             b_init=None)


    def forward(self, x0, x, edge_index, edge_weight, num_nodes):
        if self.variant:
            x = (1-self.alpha)*self.propagate(x, edge_index, edge_weight=edge_weight, num_nodes=num_nodes)
            x = (1-self.beta)*x + self.beta*self.linear(x)
            x0 = self.alpha*x0
            x0 = (1-self.beta)*x0 + self.beta * self.linear0(x0)
            x = x + x0

            # x = (1 - self.alpha) * self.propagate(x, sparse_adj)
            # x0 = self.alpha * x0
            # x = x + x0
            # x = (1-self.beta)*x + self.beta*self.linear(x)
        else:
            x = self.propagate(x, edge_index, edge_weight=edge_weight, num_nodes=num_nodes)
            x = (1-self.alpha)*x + self.alpha*x0
            x = (1-self.beta)*x + self.beta*self.linear(x)
        return x
