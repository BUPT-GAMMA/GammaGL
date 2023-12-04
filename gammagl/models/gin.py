import tensorlayerx as tlx
from .mlp import MLP
from gammagl.layers.conv.gin_conv import GINConv
from gammagl.layers.pool.glob import global_sum_pool


class GINModel(tlx.nn.Module):
    r"""The FiLM graph convolutional operator from the
        `"GNN-FiLM: Graph Neural Networks with Feature-wise Linear Modulation"
        <https://arxiv.org/abs/1906.12192>`_ paper.

        .. math::
            \mathbf{x}^{\prime}_i = \sum_{r \in \mathcal{R}}
            \sum_{j \in \mathcal{N}(i)} \sigma \left(
            \boldsymbol{\gamma}_{r,i} \odot \mathbf{W}_r \mathbf{x}_j +
            \boldsymbol{\beta}_{r,i} \right)

        Parameters
        ----------
        in_channels: int, tuple
            Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        hidden_channels: int
            Size of hidden layers of mlp.
        out_channels: int
            Size of each output sample.
        num_layers: int, optional
            Number of layers of GINConv. (default: :obj:`1`)
        name: str, optional
            name of the model.

    """

    def __init__(self, in_channels,
                 hidden_channels,
                 out_channels,
                 num_layers=4,

                 name="GIN"):
        super(GINModel, self).__init__(name=name)

        self.convs = tlx.nn.ModuleList()
        for _ in range(num_layers):
            mlp = MLP([in_channels, hidden_channels, hidden_channels])
            self.convs.append(GINConv(nn=mlp, train_eps=False))
            in_channels = hidden_channels

        self.mlp = MLP([hidden_channels, hidden_channels, out_channels],
                       norm=None, dropout=0.5)

    def forward(self, x, edge_index, batch):
        if x is None:
            # x = tlx.ones((batch.shape[0], 1), dtype=tlx.float32)
            x = tlx.random_normal((batch.shape[0], 1), dtype=tlx.float32)

        for conv in self.convs:
            x = tlx.relu(conv(x, edge_index))

        x = global_sum_pool(x, batch)
        return self.mlp(x)
