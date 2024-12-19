import tensorlayerx as tlx
from gammagl.layers.conv import GaANConv


class GaANModel(tlx.nn.Module):

    r"""The gated attentional operator from the `"Gated Attention Networks for Learning on Large and Spatiotemporal Graphs"
        <https://arxiv.org/abs/1803.07294>`_ paper.

        Parameters
        ----------
        feature_dim: int
            input feature dimension.
        hidden_dim: int
            hidden dimension.
        num_class: int
            number of classes.
        heads: int
            number of attention heads.
        drop_rate: float
            dropout rate.
        m: int, optional
            parameter to control gate value.
        v: int, optional
            parameter to control neighbor message.
        name: str, optional
            model name.

    """

    def __init__(self, feature_dim,
                 hidden_dim,
                 num_class,
                 heads,
                 drop_rate,
                 m=64,
                 v=64,
                 name=None):
        super().__init__(name=name)

        self.conv1 = GaANConv(in_channels=feature_dim,
                             out_channels=hidden_dim,
                             heads=heads,
                             m=m,
                             v=v,
                             dropout_rate=drop_rate)
        self.conv2 = GaANConv(in_channels=hidden_dim*heads,
                             out_channels=num_class,
                             heads=heads,
                             m=m,
                             v=v,
                             dropout_rate=drop_rate)
        self.elu = tlx.layers.ELU()
        self.dropout = tlx.layers.Dropout(drop_rate)

    def forward(self, x, edge_index, num_nodes):
        x = self.dropout(x)
        x = self.conv1(x, edge_index, num_nodes)
        x = self.elu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index, num_nodes)
        return x
