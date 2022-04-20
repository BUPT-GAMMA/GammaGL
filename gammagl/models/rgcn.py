import tensorlayerx as tlx
from gammagl.layers.conv import RGCNConv


class RGCN(tlx.nn.Module):
    """relational graph convoluation nerworks"""
    def __init__(self, feature_dim, hidden_dim, num_class, num_relations, name=None):
        super().__init__(name=name)

        self.conv1 = RGCNConv(feature_dim, hidden_dim, num_relations)
        self.conv2 = RGCNConv(hidden_dim, num_class, num_relations)
        self.relu = tlx.ReLU()

    def forward(self, edge_index, edge_type):
        x = self.relu(self.conv1(None, edge_index, edge_type))
        x = self.conv2(x, edge_index, edge_type)
        return x
