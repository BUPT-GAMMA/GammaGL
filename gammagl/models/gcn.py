import tensorlayerx as tl
from gammagl.layers.conv import GCNConv

class GCNModel(tl.nn.Module):
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
        self.dropout = tl.layers.Dropout(cfg.keep_rate)

    def forward(self, x, edge_index, edge_weight, num_nodes):
        x = self.conv1(x, edge_index, edge_weight, num_nodes)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_weight, num_nodes)
        return x

    @classmethod
    def calc_gcn_norm(cls, edge_index, num_nodes, edge_weight=None):
        """
        calculate GCN normilization.
        Since tf not support update value of a Tensor, we use np for calculation on CPU device.
        Args:
            edge_index: edge index
            num_nodes: number of nodes of graph
            edge_weight: edge weights of graph

        Returns:
            1-dim Tensor
        """
        import numpy as np
        import scipy.sparse as sp
        src, dst = edge_index[0], edge_index[1]
        if edge_weight is None:
            edge_weight = np.ones(edge_index.shape[1])
        A = sp.coo_matrix((edge_weight, (src, dst)))
        deg = np.sum(A, axis=1).A1
        deg_inv_sqrt = np.power(deg, -0.5)
        deg_inv_sqrt[deg_inv_sqrt==np.inf] = 0 # may exist solo node
        weights = deg_inv_sqrt[src] * edge_weight * deg_inv_sqrt[dst]
        return weights.astype(np.float32) # default is float64
