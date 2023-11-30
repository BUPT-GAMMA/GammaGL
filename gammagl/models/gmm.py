import tensorlayerx as tlx
import tensorlayerx.nn as nn
from gammagl.layers.conv import GMMConv


class GMMModel(tlx.nn.Module):
    r"""The Gaussian Mixture Model or MoNet from the `"Geometric deep learning on graphs
        and manifolds using mixture model CNNs"
        <https://arxiv.org/abs/1611.08402>`_ paper.

        Parameters
        ----------
        feature_dim: int
            input feature dimension.
        hidden_dim: int
            hidden dimension.
        num_class: int
            number of classes.
        dim: int
            Dimensionality of pseudo-coordinte.
        n_kernels: int
            Number of kernels.
        drop_rate: float
            dropout rate.
        num_layers: int
            number of layers.
        name: str, optional
            model name.

        """

    def __init__(self,
                 feature_dim,
                 hidden_dim,
                 num_class,
                 dim,
                 n_kernels,
                 drop_rate,
                 num_layers,
                 name=None,
                 ):
        super().__init__(name=name)
        self.num_layers = num_layers
        if num_layers == 1:
            self.conv = nn.ModuleList([GMMConv(feature_dim, num_class, dim=dim, n_kernels=n_kernels)])
        else:
            self.conv = nn.ModuleList([GMMConv(feature_dim, hidden_dim, dim=dim, n_kernels=n_kernels)])
            for _ in range(1, num_layers - 1):
                self.conv.append(GMMConv(hidden_dim, hidden_dim, dim=dim, n_kernels=n_kernels))
            self.conv.append(GMMConv(hidden_dim, num_class, dim=dim, n_kernels=n_kernels))

        self.tanh = tlx.Tanh()
        self.dropout = tlx.layers.Dropout(drop_rate)

    def forward(self, x, edge_index, pseudo):
        if self.num_layers == 1:
            x = self.conv[0](x, edge_index, pseudo)
            x = self.tanh(x)
        else:
            for i in range(self.num_layers - 1):
                x = self.conv[i](x, edge_index, pseudo)
                x = self.tanh(x)
                x = self.dropout(x)
            x = self.conv[-1](x, edge_index, pseudo)
            x = self.tanh(x)
        return x
