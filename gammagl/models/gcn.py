import tensorlayerx as tlx
import gammagl.mpops as mpops
from gammagl.layers.conv import GCNConv


class GCNModel(tlx.nn.Module):
    r"""
    Graph Convolutional Network proposed in `Semi-Supervised Classification with Graph Convolutional Networks`_.

    .. _Semi-Supervised Classification with Graph Convolutional Networks:
        https://arxiv.org/pdf/1609.02907.pdf
    
    Parameters
    ----------
    cfg: configuration of GCN
    """

    def __init__(self, feature_dim,
                 hidden_dim,
                 num_class,
                 drop_rate, name=None):
        super().__init__(name=name)

        self.conv1 = GCNConv(feature_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_class)
        self.relu = tlx.ReLU()
        self.dropout = tlx.layers.Dropout(drop_rate)

    def forward(self, x, edge_index, edge_weight, num_nodes):
        x = self.conv1(x, edge_index, edge_weight, num_nodes)
        x = self.relu(x)
        x = self.dropout(x)  # dropout not work in mindspore
        x = self.conv2(x, edge_index, edge_weight, num_nodes)
        return x

