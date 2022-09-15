import tensorlayerx as tlx
from gammagl.layers.conv import GCNConv


class GCNModel(tlx.nn.Module):
    r"""
    Graph Convolutional Network proposed in `Semi-Supervised Classification with Graph Convolutional Networks`_.

    .. _Semi-Supervised Classification with Graph Convolutional Networks:
        https://arxiv.org/pdf/1609.02907.pdf

    Parameters
    ----------
        feature_dim (int): input feature dimension
        hidden_dim (int): hidden dimension
        num_class (int): number of classes
        num_layers (int): number of layers
        drop_rate (float): dropout rate
        name (str): model name
    """

    def __init__(self, feature_dim,
                 hidden_dim,
                 num_class,
                 num_layers,
                 drop_rate, name=None):
        super().__init__(name=name)
        self.num_layers = num_layers
        self.convs = tlx.nn.ModuleList()
        self.dropout = tlx.layers.Dropout(drop_rate)
        self.convs.append(GCNConv(feature_dim, hidden_dim))
        if num_layers>=3:
            for i in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.convs.append(GCNConv(hidden_dim, num_class))
        else:
            self.convs.append(GCNConv(hidden_dim, num_class))
        self.relu = tlx.ReLU()


    def forward(self, x, edge_index, edge_weight, num_nodes):
        for l, layer in enumerate(self.convs):
            x = layer(x, edge_index, edge_weight, num_nodes)
            if l != len(self.convs) - 1:
                x = self.relu(x)
                x = self.dropout(x)
        return x

