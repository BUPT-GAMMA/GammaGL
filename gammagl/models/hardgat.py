import tensorlayerx as tlx
from gammagl.layers.conv import HardGATConv
from tensorlayerx.nn import ModuleList

class HardGATModel(tlx.nn.Module):
    r"""The graph hard attentional operator from the `"Graph Representation Learning via Hard and Channel-Wise Attention Networks"
    <https://dl.acm.org/doi/pdf/10.1145/3292500.3330897>`_ paper

    Parameters
    ----------
        feature_dim (int): input feature dimension
        hidden_dim (int): hidden dimension
        num_class (int): number of classes
        heads (int): number of attention heads
        drop_rate (float): dropout rate
        k (int): number of neighbors to attention
        name (str): model name

    """

    def __init__(self, feature_dim,
                 hidden_dim,
                 num_class,
                 heads,
                 drop_rate,
                 k,
                 num_layers,
                 name=None):
        super().__init__(name=name)
        self.hardgat_list = ModuleList()
        if (num_layers == 1):
            hidden_dim = num_class
        for i in range(num_layers):
            if i==0:
                self.hardgat_list.append(
                    HardGATConv(in_channels=feature_dim,
                                out_channels=hidden_dim,
                                heads=heads,
                                dropout_rate=drop_rate,
                                k=k,
                                concat=True)
                )
            elif i==(num_layers-1):
                self.hardgat_list.append(
                    HardGATConv(in_channels=hidden_dim * heads,
                                out_channels=num_class,
                                heads=heads,
                                dropout_rate=drop_rate,
                                k=k,
                                concat=False)
                )
            else:
                self.hardgat_list.append(
                    HardGATConv(in_channels=hidden_dim * heads,
                                out_channels=hidden_dim,
                                heads=heads,
                                dropout_rate=drop_rate,
                                k=k,
                                concat=True)
                )
        self.elu = tlx.layers.ELU()
        self.dropout = tlx.layers.Dropout(drop_rate)

    def forward(self, x, edge_index, num_nodes):
        for index, hardgat in enumerate(self.hardgat_list):
            x = self.dropout(x)
            x = hardgat(x, edge_index, num_nodes)
            if index < (self.hardgat_list.__len__() - 1):
                x = self.elu(x)
        return x
