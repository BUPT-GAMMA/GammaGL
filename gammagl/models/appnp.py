import tensorlayerx as tlx
from gammagl.layers.conv import APPNPConv

class APPNPModel(tlx.nn.Module):
    """
    Approximate personalized propagation of neural predictions
    """
    def __init__(self, cfg, name=None):
        super().__init__(name=name)

        # config self check
        cfg.self_check({"feature_dim": int,
                        "num_class": int,
                        "iter_K": int,
                        "alpha": float,
                        "keep_rate": float})

        self.conv = APPNPConv(cfg.feature_dim, cfg.num_class, cfg.iter_K, cfg.alpha, cfg.keep_rate)

    def forward(self, x, edge_index, edge_weight, num_nodes):
        x = self.conv(x, edge_index, edge_weight, num_nodes)
        return x
