import tensorlayer as tl
from gammagl.layers.conv import GCNConv

class GCNModel(tl.layers.Module):
    r"""
    Graph Convolutional Network proposed in `Semi-Supervised Classification with Graph Convolutional Networks`_.

    .. _Semi-Supervised Classification with Graph Convolutional Networks:
        https://arxiv.org/pdf/1609.02907.pdf
    
    Parameters:
        cfg: configuration of GCN
    """

    def __init__(self, cfg, name=None):
        super().__init__(name=name)

        # config self check
        cfg.self_check({"feature_dim": int, 
                        "hidden_dim": int, 
                        "num_class": int, 
                        "keep_rate": float})

        self.conv1 = GCNConv(cfg.feature_dim, cfg.hidden_dim)
        self.conv2 = GCNConv(cfg.hidden_dim, cfg.num_class)
        self.relu = tl.ReLU()
        self.dropout = tl.Dropout(cfg.keep_rate)

    def forward(self, x, sparse_adj):
        x = self.conv1(x, sparse_adj)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, sparse_adj)
        return x
