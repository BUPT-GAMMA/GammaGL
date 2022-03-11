import tensorlayerx as tlx
from gammagl.layers.conv import MessagePassing


class SGConv(MessagePassing):
    """simple graph convolution"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 itera_K=2):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.itera_K = itera_K

        self.linear = tlx.layers.Dense(n_units=self.out_channels,
                                      in_channels=self.in_channels,
                                      b_init=None)

    # def message_aggregate(self, x, sparse_adj):
    #     return sparse_adj @ x

    def forward(self, x, edge_index, edge_weight=None, num_nodes=None):
        x = self.linear(x)
        for _ in range(self.itera_K):
            x = self.propagate(x, edge_index, edge_weight=edge_weight, num_nodes=num_nodes)

        return x
