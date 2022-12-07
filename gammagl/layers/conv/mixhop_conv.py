import tensorlayerx as tlx
from ..conv import MessagePassing
from ...utils import degree


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
                 norm='both',
                 add_bias=True):
        super(MixHopConv, self).__init__()

        if norm not in ['left', 'right', 'none', 'both']:
            raise ValueError('Invalid norm value. Must be either "none", "both", "right" or "left".'
                             ' But got "{}".'.format(norm))

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.p = p
        self._norm = norm
        self.add_bias = add_bias

        # define weight dict for each power j
        self.weights = tlx.nn.ModuleDict({
            str(j): tlx.layers.Linear(in_features=in_channels,
                                      out_features=out_channels,
                                      W_init='xavier_uniform',
                                      b_init=None) for j in p
        })

        if add_bias:
            initor = tlx.initializers.Zeros()
            self.bias = self._get_weights("bias", shape=(1, self.out_channels * len(p)), init=initor)

    def forward(self, x, edge_index, edge_weight=None, num_nodes=None):
        src, dst = edge_index[0], edge_index[1]
        if edge_weight is None:
            edge_weight = tlx.ones(shape=(edge_index.shape[1], 1))
        weights = edge_weight
        degs = degree(dst, num_nodes=x.shape[0], dtype=tlx.float32)
        if self._norm in ['left', 'both']:
            if self._norm == 'both':
                norm = tlx.pow(degs, -0.5)
            else:
                norm = 1.0 / degs
            weights = tlx.ops.gather(norm, src) * tlx.reshape(edge_weight, (-1,))
        degs = degree(dst, num_nodes=x.shape[0], dtype=tlx.float32)
        if self._norm in ['right', 'both']:
            if self._norm == 'both':
                norm = tlx.pow(degs, -0.5)
            else:
                norm = 1.0 / degs
            weights = tlx.reshape(weights, (-1,)) * tlx.ops.gather(norm, dst)

        max_j = max(self.p) + 1
        outputs = []
        for j in range(max_j):
            if j in self.p:
                output = self.weights[f'{j}'](x)
                outputs.append(output)

            x = self.propagate(x, edge_index, edge_weight=weights, num_nodes=num_nodes)

        final = tlx.concat(outputs, axis=1)

        if self.add_bias:
            final += self.bias

        return final
