import tensorlayer as tl
from gammagl.layers.conv import APPNPConv

class APPNPModel(tl.layers.Module):
    """
    Approximate personalized propagation of neural predictions
    """
    def __init__(self, cfg, name=None):
        super().__init__(name=name)

        # config self check
        cfg.self_check({"feature_dim": int,
                        "num_class": int,
                        "itera_K": int,
                        "alpha": float,
                        "keep_rate": float})

        self.conv = APPNPConv(cfg.feature_dim, cfg.num_class, cfg.itera_K, cfg.alpha, cfg.keep_rate)

    def forward(self, x, sparse_adj):
        x = self.conv(x, sparse_adj)
        return x