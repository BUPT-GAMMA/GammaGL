import tensorflow as tf
import tensorlayer as tl
from gammagl.layers.conv import MessagePassing

class APPNPConv(MessagePassing):
    '''
    Approximate personalized propagation of neural predictions
    '''
    def __init__(self,
                 in_channels,
                 out_channels,
                 itera_K,
                 alpha,
                 keep_rate):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.itera_K = itera_K
        self.alpha = alpha
        self.keep_rate = keep_rate

        self.linear_w = tl.layers.Dense(n_units=self.out_channels,
                                        in_channels=self.in_channels,
                                        b_init=None)
        self.dropout = tl.Dropout(self.keep_rate)

    def message_aggregate(self, x, sparse_adj):
        return sparse_adj @ x

    def forward(self, x, sparse_adj, edge_weight=None):
        x = self.linear(x)
        h = x
        for _ in range(self.itera_K):
            x = self.propagate(x, sparse_adj)
            x = x * (1 - self.alpha)
            x += self.alpha * h
        return x
