"""CoED directional convolution layer.

This module implements the directional message passing operator used in
`"Co-Embedding of Edges and Directions for Graph Neural Networks"
<https://arxiv.org/abs/2410.14109>`_.
"""

import tensorlayerx as tlx
from tensorlayerx.nn import Linear

from gammagl.layers.conv import MessagePassing


class CoEDConv(MessagePassing):
    r"""The directional convolution operator used by CoED-GNN.

    The layer separately aggregates messages for two directional channels and
    optionally applies an additional self-feature transformation.

    Parameters
    ----------
    in_channels: int
        Size of each input sample.
    out_channels: int
        Size of each output sample.
    self_feature_transform: bool, optional
        If set to :obj:`True`, adds an extra linear transform on the input node
        features and combines it with directional messages.
    bias: bool, optional
        If set to :obj:`False`, the layer will not learn additive bias terms.

    """

    def __init__(self, in_channels, out_channels, self_feature_transform=True, bias=True):
        super().__init__()
        self.self_feature_transform = self_feature_transform

        self.lin_src_to_dst = Linear(
            in_features=in_channels,
            out_features=out_channels,
            W_init="xavier_uniform",
            b_init=None,
        )
        self.lin_dst_to_src = Linear(
            in_features=in_channels,
            out_features=out_channels,
            W_init="xavier_uniform",
            b_init=None,
        )

        if self_feature_transform:
            self.lin_self = Linear(
                in_features=in_channels,
                out_features=out_channels,
                W_init="xavier_uniform",
                b_init=None,
            )
        else:
            self.lin_self = None

        if bias:
            zeros = tlx.initializers.Zeros()
            self.bias_src_to_dst = self._get_weights("bias_src_to_dst", shape=(out_channels,), init=zeros)
            self.bias_dst_to_src = self._get_weights("bias_dst_to_src", shape=(out_channels,), init=zeros)
            self.bias_self = (
                self._get_weights("bias_self", shape=(out_channels,), init=zeros)
                if self_feature_transform
                else None
            )
        else:
            self.bias_src_to_dst = None
            self.bias_dst_to_src = None
            self.bias_self = None

    def forward(self, x, edge_index, edge_weight=None, num_nodes=None):
        """Compute directional node representations."""
        if num_nodes is None:
            num_nodes = tlx.get_tensor_shape(x)[0]

        if isinstance(edge_weight, (tuple, list)):
            edge_weight_src_to_dst, edge_weight_dst_to_src = edge_weight
        else:
            edge_weight_src_to_dst = edge_weight
            edge_weight_dst_to_src = edge_weight

        x_src_to_dst = self.propagate(
            x=x,
            edge_index=edge_index,
            edge_weight=edge_weight_src_to_dst,
            num_nodes=num_nodes,
        )
        x_dst_to_src = self.propagate(
            x=x,
            edge_index=edge_index,
            edge_weight=edge_weight_dst_to_src,
            num_nodes=num_nodes,
        )

        x_src_to_dst = self.lin_src_to_dst.forward(x_src_to_dst)
        x_dst_to_src = self.lin_dst_to_src.forward(x_dst_to_src)

        if self.bias_src_to_dst is not None:
            x_src_to_dst = x_src_to_dst + self.bias_src_to_dst
        if self.bias_dst_to_src is not None:
            x_dst_to_src = x_dst_to_src + self.bias_dst_to_src

        if self.self_feature_transform:
            x_self = self.lin_self.forward(x)
            if self.bias_self is not None:
                x_self = x_self + self.bias_self
            return x_src_to_dst, x_dst_to_src, x_self

        return x_src_to_dst, x_dst_to_src

    def message(self, x, edge_index, edge_weight=None):
        """Construct messages on each edge."""
        msg = tlx.gather(x, edge_index[0, :])
        if edge_weight is None:
            return msg
        return msg * tlx.reshape(edge_weight, (-1, 1))
