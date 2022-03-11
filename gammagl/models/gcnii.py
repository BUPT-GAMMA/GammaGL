import math
import tensorlayerx as tlx
from gammagl.layers.conv import GCNIIConv

class GCNIIModel(tlx.nn.Module):
    """GCN with Initial residual and Identity mapping"""
    def __init__(self, cfg, name=None):
        super().__init__(name=name)

        # config self check
        cfg.self_check({"feature_dim": int, 
                        "hidden_dim": int, 
                        "num_class": int,
                        "num_layers": int,
                        "alpha": float,
                        "beta": float,
                        "lambd": float, 
                        "variant": bool,
                        "keep_rate": float})

        self.linear_head = tlx.layers.Dense(n_units=cfg.hidden_dim,
                                      in_channels=cfg.feature_dim,
                                      b_init=None)
        self.linear_tail = tlx.layers.Dense(n_units=cfg.num_class,
                                      in_channels=cfg.hidden_dim,
                                      b_init=None)
    
        self.convs = tlx.layers.SequentialLayer()
        for i in range(1, cfg.num_layers+1):
            # cfg.beta = cfg.lambd/i if cfg.variant else cfg.beta
            cfg.beta = math.log(cfg.lambd/i+1) if cfg.variant else cfg.beta
            self.convs.append(GCNIIConv(cfg.hidden_dim, cfg.hidden_dim, cfg.alpha, cfg.beta, cfg.variant))

        self.relu = tlx.ReLU()
        self.dropout = tlx.layers.Dropout(cfg.keep_rate)

    def forward(self, x, edge_index, edge_weight, num_nodes):
        x0 = x = self.relu(self.linear_head(self.dropout(x)))
        for conv in self.convs:
            x = self.dropout(x)
            x = conv(x0, x, edge_index, edge_weight, num_nodes)
            x = self.relu(x)
        x = self.relu(self.linear_tail(self.dropout(x)))
        return x
