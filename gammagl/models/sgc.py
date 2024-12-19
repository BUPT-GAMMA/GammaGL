import tensorlayerx as tlx
from gammagl.layers.conv import SGConv


class SGCModel(tlx.nn.Module):
    """simplifing graph convoluation nerworks"""
    def __init__(self, feature_dim, num_class, iter_K, name=None):
        super().__init__(name=name)

        self.conv = SGConv(feature_dim, num_class, iter_K=iter_K)

    def forward(self, x, edge_index, edge_weight, num_nodes):
        x = self.conv(x, edge_index, edge_weight, num_nodes)
        return x
