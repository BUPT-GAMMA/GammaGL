import tensorlayerx as tlx
from gammagl.layers.conv import GATV2Conv
import tensorflow as tf

class GATV2Model(tlx.nn.Module):
    r"""`"How Attentive are Graph Attention Networks?"
    <https://arxiv.org/abs/2105.14491>`_ paper.

    Parameters:
        cfg: configuration of GAT
    """

    def __init__(self, cfg, name=None):
        super().__init__(name=name)

        # config self check
        cfg.self_check({"feature_dim": int,
                        "hidden_dim": int,
                        "heads": int,
                        "num_class": int,
                        "keep_rate": float})

        self.conv1 = GATV2Conv(in_channels=cfg.feature_dim,
                             out_channels=cfg.hidden_dim,
                             heads=cfg.heads,
                             dropout_rate=1-cfg.keep_rate,
                             concat=True)
        self.conv2 = GATV2Conv(in_channels=cfg.hidden_dim*cfg.heads,
                             out_channels=cfg.num_class,
                             heads=cfg.heads,
                             dropout_rate=1-cfg.keep_rate,
                             concat=False)
        self.elu = tlx.layers.ELU()
        self.dropout = tlx.layers.Dropout(cfg.keep_rate)

    def forward(self, x, edge_index, num_nodes):
        x = self.dropout(x)
        x = self.conv1(x, edge_index, num_nodes)
        x = self.elu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index, num_nodes)
        return x
