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
    in_channels(int): int
        Size of each input sample.
    out_channels(int): int
        Size of each output sample.
    """
    def __init__(self, in_channels, out_channels, kernel_size = 1, negative_slope = 0.2):
        super(EdgeConv, self).__init__()
        self.nn = tlx.nn.Sequential(
            tlx.layers.Conv2d(2 * in_channels, out_channels, kernel_size=kernel_size, b_init=None),
            tlx.layers.BatchNorm2d(out_channels),
            tlx.LeakyReLU(negative_slope=negative_slope)
        )

    def message(self, x_i, x_j, edge_weight=None):
        tmp = tlx.concat([x_i, x_j - x_i], axis=-1)
        return self.nn(tmp)

    def forward(self, x, edge_index):
        return self.propagate(x, edge_index, aggr='max')
