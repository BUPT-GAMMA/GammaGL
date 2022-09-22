import tensorlayerx as tlx
import tensorlayerx.nn as nn
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
        drop_rate (float): dropout rate
        num_layers (int): number of layers
        norm (str): apply the normalizer
        name (str): model name
    """

    def __init__(self, feature_dim,
                 hidden_dim,
                 num_class,
                 drop_rate,
                 num_layers = 2,
                 norm = 'both',
                 name=None):
        super().__init__(name=name)
        if num_layers < 2:
            raise ValueError('The number of layers should be at least 2'
                             ' But got "{}".'.format(num_layers))
        self.num_layers = num_layers
        self.conv = nn.ModuleList([GCNConv(feature_dim, hidden_dim, norm = norm)])
        for _ in range(1, num_layers - 1):
            self.conv.append(GCNConv(hidden_dim, hidden_dim, norm = norm))
        self.conv.append(GCNConv(hidden_dim, num_class, norm = 'none'))
        # self.conv1 = GCNConv(feature_dim, hidden_dim, norm = 'both')
        # self.conv2 = GCNConv(hidden_dim, num_class, norm = 'both')
        self.relu = tlx.ReLU()
        self.dropout = tlx.layers.Dropout(drop_rate)

    def forward(self, x, edge_index, edge_weight, num_nodes):
        for i in range(self.num_layers - 1):
            x = self.conv[i](x, edge_index, edge_weight, num_nodes)
            x = self.relu(x)
            x = self.dropout(x)
        x = self.conv[-1](x, edge_index, edge_weight, num_nodes)
        # x = self.conv1(x, edge_index, edge_weight, num_nodes)
        # x = self.relu(x)
        # x = self.dropout(x)  # dropout not work in mindspore
        # x = self.conv2(x, edge_index, edge_weight, num_nodes)
        return x

