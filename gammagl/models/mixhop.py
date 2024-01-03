import tensorlayerx as tlx
from ..layers.conv import MixHopConv


class MIXHOPModel(tlx.nn.Module):
    r"""MixHop proposed in `"MixHop: Higher-Order Graph Convolutional
    Architectures via Sparsified Neighborhood Mixing"
    <https://arxiv.org/abs/1905.00067>`_ paper

    Parameters
    ----------
        feature_dim: int
            input feature dimension
        hidden_dim: int
            hidden dimension
        out_dim: int
            The number of classes for prediction
        p: list
            The list of integer adjacency powers
        drop_rate: float
            dropout rate
        num_layers: int
            Number of Mixhop Graph Convolutional Layers
        name: str
            model name
    """

    def __init__(self, feature_dim,
                 hidden_dim,
                 out_dim,
                 p,
                 drop_rate,
                 num_layers=3,
                 norm='both',
                 name=None):
        super(MIXHOPModel, self).__init__(name=name)
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.p = p
        self.drop_rate = drop_rate
        self.dropout = tlx.layers.Dropout(self.drop_rate)
        self.relu = tlx.ReLU()

        # Input layer
        self.layer_head = MixHopConv(in_channels=self.feature_dim,
                                     out_channels=self.hidden_dim,
                                     p=self.p,
                                     norm=norm)

        self.layers_list = []
        for i in range(1, self.num_layers - 1):
            self.layers_list.append(
                MixHopConv(in_channels=self.hidden_dim * len(p), out_channels=self.hidden_dim, p=self.p, norm=norm))
        self.layers = tlx.nn.ModuleList(self.layers_list)

        W_initor = tlx.initializers.XavierUniform()
        self.fc_layers = tlx.layers.Linear(in_features=self.hidden_dim * len(self.p), W_init=W_initor,
                                           out_features=self.out_dim)

    def forward(self, x, edge_index, edge_weight, num_nodes=None):
        x = self.layer_head(self.dropout(x), edge_index, num_nodes=num_nodes)
        x = self.relu(x)

        for i in range(len(self.layers_list)):
            x = self.dropout(x)
            x = self.layers_list[i](x, edge_index, edge_weight, num_nodes)
            x = self.relu(x)

        x = self.fc_layers(x)
        return x
