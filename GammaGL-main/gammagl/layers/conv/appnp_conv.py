import tensorlayerx as tlx
from gammagl.layers.conv import MessagePassing

class APPNPConv(MessagePassing):
    '''
    Approximate personalized propagation of neural predictions
    '''
    def __init__(self,
                 in_channels,
                 out_channels,
                 iter_K,
                 alpha,
                 drop_rate):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.iter_K = iter_K
        self.alpha = alpha
        self.drop_rate = drop_rate

        self.linear_w = tlx.nn.layers.Linear(out_features=out_channels,
                                             in_features=in_channels,
                                             b_init=None)
        self.dropout = tlx.nn.layers.Dropout(self.drop_rate)
        self.dropout_e = tlx.nn.layers.Dropout(0.9)

    def forward(self, x, edge_index, edge_weight=None, num_nodes=None):
        x = self.linear_w(x)
        h0 = x
        x = self.dropout(x)
        for _ in range(self.iter_K):
            if edge_weight is not None:
                edge_weight = self.dropout_e(edge_weight)
            x = self.propagate(x, edge_index, edge_weight=edge_weight, num_nodes=num_nodes)
            x = x * (1 - self.alpha)
            x += self.alpha * h0
        return x
