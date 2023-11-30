import os
os.environ['TL_BACKEND'] = 'torch'
import tensorlayerx as tlx
from gammagl.layers.conv import GATConv
from tensorlayerx.nn import ModuleList
class GATModel(tlx.nn.Module):

    r"""The graph attentional operator from the `"Graph Attention Networks"
        <https://arxiv.org/abs/1710.10903>`_ paper.

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
        name: str, optional
            model name.

    """

    def __init__(self,feature_dim,
                 hidden_dim,
                 num_class,
                 heads,
                 drop_rate,
                 num_layers,
                 name=None,
                 ):
        super().__init__(name=name)

        self.gat_list = ModuleList()
        if(num_layers==1):
            hidden_dim = num_class
        for i in range(num_layers):
            if i==0:
                self.gat_list.append(
                    GATConv(in_channels=feature_dim,
                             out_channels=hidden_dim,
                             heads=heads,
                             dropout_rate=drop_rate,
                             concat=True))
            elif i==(num_layers-1):
                self.gat_list.append(
                    GATConv(in_channels=hidden_dim * heads,
                            out_channels=num_class,
                            heads=heads,
                            dropout_rate=drop_rate,
                            concat=False))

            else:
                self.gat_list.append(
                    GATConv(in_channels=hidden_dim * heads,
                            out_channels=hidden_dim,
                            heads=heads,
                            dropout_rate=drop_rate,
                            concat=True))

        self.elu = tlx.layers.ELU()
        self.dropout = tlx.layers.Dropout(drop_rate)

    def forward(self, x, edge_index, num_nodes):
        for index,gat in enumerate(self.gat_list):
            x = self.dropout(x)
            x = gat(x,edge_index,num_nodes)
            if index<(self.gat_list.__len__()-1):
                x = self.elu(x)

        return x
