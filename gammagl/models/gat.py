import tensorlayerx as tlx
from gammagl.layers.conv import GATConv

class GATModel(tlx.nn.Module):

    r"""The graph attentional operator from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper

    Parameters
    ----------
        feature_dim (int): input feature dimension
        hidden_dim (int): hidden dimension
        num_class (int): number of classes
        heads (int): number of attention heads
        drop_rate (float): dropout rate
        name (str): model name

    """

    def __init__(self,feature_dim,
                 hidden_dim,
                 num_class,
                 heads,
                 drop_rate,
                 name=None):
        super().__init__(name=name)


        self.conv1 = GATConv(in_channels=feature_dim,
                             out_channels=hidden_dim,
                             heads=heads,
                             dropout_rate=drop_rate,
                             concat=True)
        self.conv2 = GATConv(in_channels=hidden_dim*heads,
                             out_channels=num_class,
                             heads=heads,
                             dropout_rate=drop_rate,
                             concat=False)
        self.elu = tlx.layers.ELU()
        self.dropout = tlx.layers.Dropout(drop_rate)

    def forward(self, x, edge_index, num_nodes):
        x = self.dropout(x)
        x = self.conv1(x, edge_index, num_nodes)
        x = self.elu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index, num_nodes)
        return x
