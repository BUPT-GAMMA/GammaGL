import tensorlayerx as tlx
from ..conv import MessagePassing
from ... import utils


class MixHopConv(MessagePassing):
    r"""The sparsified neighborhood mixing graph convolutional operator from the
    `"MixHop: Higher-Order Graph Convolutional Architectures
    via Sparsified Neighborhood Mixing"
    <https://arxiv.org/abs/1905.00067>`_ paper

    .. math::
        \mathit{H^{(i+1)}}=\Vert_{{j\in P}}{\sigma ({\widehat{A}^{j}H^{(i)}W_{j}^{(i)}})},

    where :math:`\hat{A} = D^{-\tfrac{1}{2}}(A + I_n)D^{-\tfrac{1}{2}}` is a symmetrically
    normalized adjacency matrix with self-connections and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.

    Parameters
    ----------
        in_channels: int
            Size of each input sample
        out_channels: int
            Size of each output sample
        p: list
            The list of integer adjacency powers
        add_bias: bool
            If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 p,
                 add_bias=True):
        super(MixHopConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.p = p
        self.add_bias = add_bias

        # define weight dict for each power j
        self.weights = tlx.nn.ModuleDict({
            str(j): tlx.layers.Linear(in_features=in_channels,
                                      out_features=out_channels,
                                      W_init='xavier_uniform',
                                      b_init=None) for j in p
        })

        if add_bias is True:
            initor = tlx.initializers.Zeros()
            self.bias = self._get_weights("bias", shape=(1, self.out_channels * len(p)), init=initor)

    def forward(self, x, edge_index, edge_weight=None, num_nodes=None):
        row, col = edge_index
        degs = utils.degree(col, dtype=tlx.float32)
        norm = tlx.pow(degs, -0.5)
        norm = tlx.reshape(norm, (x.shape[0], 1))

        max_j = max(self.p) + 1
        outputs = []
        for j in range(max_j):
            if j in self.p:
                output = self.weights[f'{j}'](x)
                outputs.append(output)

            x = x * norm
            x = self.propagate(x, edge_index, edge_weight=None, num_nodes=num_nodes)
            x = x * norm

        final = tlx.concat(outputs, axis=1)

        if self.add_bias:
            final += self.bias

        return final
