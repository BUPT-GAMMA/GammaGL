import math
import tensorlayerx as tlx
from gammagl.layers.conv import GCNIIConv


class GCNIIModel(tlx.nn.Module):
    """
    The graph convolutional operator with initial residual connections and
    identity mapping (GCNII) from the `"Simple and Deep Graph Convolutional
    Networks" <https://arxiv.org/abs/2007.02133>`_ paper
    """

    def __init__(self, feature_dim, hidden_dim, num_class, num_layers,
                 alpha, beta, lambd, variant, drop_rate, name=None):
        super().__init__(name=name)

        self.linear_head = tlx.layers.Linear(out_features=hidden_dim,
                                             in_features=feature_dim,
                                             b_init=None)
        self.linear_tail = tlx.layers.Linear(in_features=hidden_dim,
                                             out_features=num_class,
                                             b_init=None)

        self.convs = tlx.nn.ModuleList([])
        for i in range(1, num_layers + 1):
            # beta = lambd / i if variant else beta
            beta = math.log(lambd / i + 1) if variant else beta
            self.convs.append(GCNIIConv(hidden_dim, hidden_dim, alpha, beta, variant))

        self.relu = tlx.ReLU()
        self.dropout = tlx.layers.Dropout(drop_rate)

    def forward(self, x, edge_index, edge_weight, num_nodes):
        x0 = x = self.relu(self.linear_head(self.dropout(x)))
        for conv in self.convs:
            x = self.dropout(x)
            x = conv(x0, x, edge_index, edge_weight, num_nodes)
            x = self.relu(x)
        x = self.relu(self.linear_tail(self.dropout(x)))
        return x
