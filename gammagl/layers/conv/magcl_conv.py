import math
import tensorlayerx as tlx
from gammagl.utils import degree
from gammagl.layers.conv import MessagePassing

class MAGCLConv(MessagePassing):
    r"""The graph convolutional operator from the `"MA-GCL: Model Augmentation
    Tricks for Graph Contrastive Learning"
    <https://arxiv.org/abs/1609.02907>`_ paper

    we focus on analyzing graph signals and assume one-hot node features
    :math:`\boldsymbol{X}=\boldsymbol{I},`
    Then the embedding of two views can be written as
    :math:`\boldsymbol{Z}=\boldsymbol{U} \Lambda^{L} \boldsymbol{U}^{T} \boldsymbol{W}` ,
    :math:`\boldsymbol{Z'}=\boldsymbol{U} \Lambda^{L'} \boldsymbol{U}^{T} \boldsymbol{W}`, where
    :math:`\boldsymbol{W} \in \mathbb{R}^{|\mathcal{V}| \times d_{O}}`

    the loss is written as:

    .. math::
        \min _{\boldsymbol{W}} =
        \sum_{i=1}^{|\mathcal{V}|}\left|\boldsymbol{z}_{i}-\boldsymbol{z}_{i}^{\prime}\right|^{2} =
        \min _{\boldsymbol{W}} \text{tr}\left(\left(\boldsymbol{Z}-\boldsymbol{Z}^{\prime}\right)\left(\boldsymbol{Z}-\boldsymbol{Z}^{\prime}\right)^{T}\right)


    subject to :math:`\boldsymbol{W}^{T} \boldsymbol{W}=\boldsymbol{I}`.

    Parameters
    ----------
    in_channels: int
        Size of each input sample
    out_channels: int
        Size of each output sample.
    norm: str, optional
        How to apply the normalizer.  Can be one of the following values:

            * ``right``, to divide the aggregated messages by each node's in-degrees, which is equivalent to averaging the received messages.

            * ``none``, where no normalization is applied.

            * ``both`` (default), where the messages are scaled with :math:`1/c_{ji}` above, equivalent to symmetric normalization.

            * ``left``, to divide the messages sent out from each node by its out-degrees, equivalent to random walk normalization.
    add_bias: bool, optional
        If set to :obj:`False`, the layer will not learn
        an additive bias. (default: :obj:`True`)

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 norm='both',
                 add_bias=True):
        super().__init__()

        if norm not in ['left', 'right', 'none', 'both']:
            raise ValueError('Invalid norm value. Must be either "none", "both", "right" or "left".'
                             ' But got "{}".'.format(norm))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.add_bias = add_bias
        self._norm = norm
        initor = tlx.initializers.TruncatedNormal()
        # stdv = math.sqrt(6.0 / (in_channels + out_channels))
        # self.weight = tlx.nn.Parameter(tlx.random_uniform((in_channels, out_channels), -stdv, stdv))
        self.weight = self._get_weights("weight", shape=(in_channels, out_channels), init=initor, order=True)
        if add_bias is True:
            initor = tlx.initializers.Zeros()
            self.bias = self._get_weights("bias", shape=(1, self.out_channels), init=initor)

    def forward(self, x, edge_index, k, edge_weight=None, num_nodes=None):
        x = tlx.matmul(x, self.weight)

        src, dst = edge_index[0], edge_index[1]
        if edge_weight is None:
            edge_weight = tlx.ones(shape=(edge_index.shape[1], 1))
        edge_weight = tlx.reshape(edge_weight, (-1,))
        weights = edge_weight
        deg = degree(src, num_nodes=x.shape[0], dtype=tlx.float32)
        norm = tlx.pow(deg, -0.5)
        weights = tlx.ops.gather(norm, src) * tlx.reshape(edge_weight, (-1,))
        weights = tlx.reshape(weights, (-1,)) * tlx.ops.gather(norm, dst)

        out = self.propagate(x, edge_index, edge_weight=weights, num_nodes=num_nodes)
        for _ in range(k - 1):
            out = out + self.propagate(x, edge_index, edge_weight=weights, num_nodes=num_nodes)
            out = 0.5 * out

        if self.add_bias:
            out += self.bias
        return out
