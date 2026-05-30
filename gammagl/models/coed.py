"""CoED-GNN backbone model.

This module implements the node classification backbone described in
`"Co-Embedding of Edges and Directions for Graph Neural Networks"
<https://arxiv.org/abs/2410.14109>`_.
"""

import tensorlayerx as tlx
from tensorlayerx.nn import Dropout, Linear, Module, ReLU

from gammagl.layers.conv import CoEDConv, JumpingKnowledge


class CoEDModel(Module):
    r"""CoED-GNN model for node classification.

    Parameters
    ----------
    feature_dim: int
        Input feature dimension.
    hidden_dim: int
        Hidden feature dimension.
    num_class: int
        Number of output classes.
    num_layers: int, optional
        Number of directional convolution layers.
    alpha: float, optional
        Mixture coefficient for combining the two directional channels.
    drop_rate: float, optional
        Dropout rate applied between hidden layers.
    normalize: bool, optional
        If set to :obj:`True`, applies L2 normalization to hidden features.
    self_feature_transform: bool, optional
        If set to :obj:`True`, each CoED layer also learns a self-feature
        transform branch.
    jumping_knowledge: str, optional
        Type of jumping-knowledge aggregation (:obj:`"cat"`, :obj:`"max"`,
        :obj:`"lstm"`, or :obj:`None`).  When set, intermediate layer
        outputs are aggregated and projected through an additional linear
        layer.
    name: str, optional
        Model name.

    """

    def __init__(
        self,
        feature_dim,
        hidden_dim,
        num_class,
        num_layers=2,
        alpha=0.0,
        drop_rate=0.5,
        normalize=False,
        self_feature_transform=False,
        jumping_knowledge=None,
        name=None,
    ):
        super().__init__(name=name)
        self.alpha = alpha
        self.num_layers = num_layers
        self.normalize = normalize
        self.jumping_knowledge = jumping_knowledge

        self.convs = []
        in_channels = feature_dim
        for layer_idx in range(num_layers):
            conv = CoEDConv(
                in_channels=in_channels,
                out_channels=hidden_dim,
                self_feature_transform=self_feature_transform,
            )
            self.convs.append(conv)
            self.add_module("conv{}".format(layer_idx + 1), conv)
            in_channels = hidden_dim

        if jumping_knowledge is not None:
            self.jump = JumpingKnowledge(jumping_knowledge, hidden_dim, num_layers)
            if jumping_knowledge == "cat":
                jk_dim = hidden_dim * num_layers
            else:
                jk_dim = hidden_dim
            self.lin = Linear(
                in_features=jk_dim,
                out_features=num_class,
                W_init="xavier_uniform",
                b_init=tlx.initializers.Zeros(),
            )
            self.readout = None
        else:
            self.jump = None
            self.lin = None
            self.readout = Linear(
                in_features=hidden_dim,
                out_features=num_class,
                W_init="xavier_uniform",
                b_init=tlx.initializers.Zeros(),
            )

        self.relu = ReLU()
        self.dropout = Dropout(p=drop_rate)

    def combine(self, xs):
        """Combine directional features with the optional self-feature branch."""
        if len(xs) == 3:
            x_src_to_dst, x_dst_to_src, x_self = xs
            return self.alpha * x_src_to_dst + (1.0 - self.alpha) * x_dst_to_src + x_self

        x_src_to_dst, x_dst_to_src = xs
        return self.alpha * x_src_to_dst + (1.0 - self.alpha) * x_dst_to_src

    def forward(self, x, edge_index, edge_weight=None, num_nodes=None):
        """Compute node logits."""
        x_intermediate = []

        for layer_idx, conv in enumerate(self.convs):
            x = self.combine(conv.forward(x, edge_index, edge_weight=edge_weight, num_nodes=num_nodes))

            if layer_idx != self.num_layers - 1 or self.jump is not None:
                x = self.relu.forward(x)
                x = self.dropout.forward(x)
                if self.normalize:
                    x = tlx.l2_normalize(x, axis=1)
                x_intermediate.append(x)

        if self.jump is not None:
            x = self.jump(x_intermediate)
            x = self.lin.forward(x)
        else:
            x = self.readout.forward(x)

        return x
