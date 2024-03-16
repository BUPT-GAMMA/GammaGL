import os
os.environ['TL_BACKEND'] = 'torch'
import tensorlayerx as tlx
from gammagl.layers.conv import FusedGATConv

class FusedGATModel(tlx.nn.Module):

    r"""The graph attentional operator from the `"Understanding GNN Computational Graph: A Coordinated Computation, IO, and Memory Perspective"
    <https://arxiv.org/abs/2110.09524`_ paper.

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

        self.fusedgat_list = tlx.nn.ModuleList()
        if num_layers == 1:
            hidden_dim = num_class
        for i in range(num_layers):
            if i == 0:
                self.fusedgat_list.append(
                    FusedGATConv(in_channels=feature_dim,
                                 out_channels=hidden_dim,
                                 heads=heads,
                                 dropout_rate=drop_rate,
                                 concat=True))

            elif i == num_layers - 1:
                self.fusedgat_list.append(
                    FusedGATConv(in_channels=hidden_dim * heads,
                                 out_channels=num_class,
                                 heads=heads,
                                 dropout_rate=drop_rate,
                                 concat=False))

            else:
                self.fusedgat_list.append(
                    FusedGATConv(in_channels=hidden_dim * heads,
                                 out_channels=hidden_dim,
                                 heads=heads,
                                 dropout_rate=drop_rate,
                                 concat=True))

        self.elu = tlx.layers.ELU()
        self.dropout = tlx.layers.Dropout(drop_rate)

    def forward(self, x, edge_index, num_nodes, **kwargs):
        for index, fusedgat in enumerate(self.fusedgat_list):
            x = self.dropout(x)
            x = fusedgat(x, edge_index, num_nodes, **kwargs)
            if index < self.fusedgat_list.__len__() - 1:
                x = self.elu(x)

        return x