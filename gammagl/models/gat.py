import tensorlayer as tl
from gammagl.layers import GATConv

class GATModel(tl.layers.Module):
    r"""The graph attentional operator from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper

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

        self.conv1 = GATConv(in_channels=cfg.feature_dim,
                             out_channels=cfg.hidden_dim,
                             heads=cfg.heads,
                             dropout_rate=1-cfg.keep_rate,
                             concat=True)
        self.conv2 = GATConv(in_channels=cfg.hidden_dim*cfg.heads,
                             out_channels=cfg.num_class,
                             heads=cfg.heads,
                             dropout_rate=1-cfg.keep_rate,
                             concat=False)
        self.elu = tl.layers.ELU()
        self.dropout = tl.Dropout(cfg.keep_rate)

    def forward(self, x, edge_index, num_nodes):
        x = self.dropout(x)
        x = self.conv1(x, edge_index, num_nodes)
        x = self.elu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index, num_nodes)
        return x
