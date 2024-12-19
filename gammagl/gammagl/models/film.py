import tensorlayerx as tlx
from gammagl.layers.conv.film_conv import FILMConv


class FILMModel(tlx.nn.Module):
    r"""The FiLM graph convolutional operator from the
        `"GNN-FiLM: Graph Neural Networks with Feature-wise Linear Modulation"
        <https://arxiv.org/abs/1906.12192>`_ paper.

        Parameters
        ----------
        in_channels: int, tuple
            Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        hidden_dim: int
            Size of each hidden layers sample.
        out_channels: int
            Size of each output sample.
        drop_rate: float
            Rate of dropout layers.
        name: str, optional
            Name of the model.

    """

    def __init__(self, in_channels,
                 hidden_dim,
                 out_channels,
                 num_layers,
                 drop_rate,
                 name=None):
        super(FILMModel, self).__init__(name=name)

        self.convs = tlx.nn.ModuleList()
        self.convs.append(FILMConv(in_channels=in_channels, out_channels=hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(FILMConv(in_channels=hidden_dim, out_channels=hidden_dim))
        self.convs.append(FILMConv(in_channels=hidden_dim, out_channels=out_channels, act=None))

        self.norms = tlx.nn.ModuleList()
        for _ in range(num_layers - 1):
            self.norms.append(tlx.nn.BatchNorm1d(num_features=hidden_dim, momentum=0.1,gamma_init='ones'))
        self.dropout = tlx.nn.Dropout(drop_rate)

    def forward(self, x, edge_index):
        for conv, norm in zip(self.convs[:-1], self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = self.dropout(x)
        x = self.convs[-1](x, edge_index)
        return x