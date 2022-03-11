import tensorlayerx as tl
from gammagl.layers.conv import SGConv

class SGCModel(tl.nn.Module):
    """simplifing graph convoluation nerworks"""
    def __init__(self, cfg, name=None):
        super().__init__(name=name)

        # config self check
        cfg.self_check({"feature_dim": int,
                        "num_class": int,
                        "itera_K": int})

        self.conv = SGConv(cfg.feature_dim, cfg.num_class, itera_K=cfg.itera_K)

    def forward(self, x, edge_index, edge_weight, num_nodes):
        x = self.conv(x, edge_index, edge_weight, num_nodes)
        return x
