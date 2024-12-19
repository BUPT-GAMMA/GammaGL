import tensorlayerx as tlx
from gammagl.layers.conv import APPNPConv

class APPNPModel(tlx.nn.Module):
    """
    Approximate personalized propagation of neural predictions
    """
    def __init__(self, feature_dim, num_class, iter_K,
                 alpha, drop_rate, name=None):
        super().__init__(name=name)

        self.conv = APPNPConv(feature_dim, num_class, iter_K, alpha, drop_rate)

    def forward(self, x, edge_index, edge_weight, num_nodes):
        x = self.conv(x, edge_index, edge_weight, num_nodes)
        return x
