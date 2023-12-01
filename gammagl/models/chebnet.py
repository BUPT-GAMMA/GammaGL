import numpy as np
import tensorlayerx as tlx
from scipy.sparse import linalg, diags, identity, csr, csr_matrix
from tensorlayerx.nn import Module

from gammagl.layers.conv.cheb_conv import ChebConv


class ChebNetModel(Module):
    r"""Graph Convolutional Network proposed in `"Convolutional Neural Networks on Graphs with Fast Localized Spectral
        Filtering" <https://arxiv.org/abs/1606.09375>`_ paper.

        Parameters
        ----------
        feature_dim: int
            The dimensionality of input feature.
        hidden_dim: int
            The dimensionality of hidden layer.
        out_dim: int
            The number of classes for prediction.
        k: int
            Chebyshev filter size.
        drop_rate: float
            Dropout rate.
        name: str
            The name of the model.

        """

    def __init__(self, feature_dim, hidden_dim, out_dim, k, drop_rate, name=None):
        super().__init__()
        self.conv1 = ChebConv(feature_dim, hidden_dim, K=k)
        self.conv2 = ChebConv(hidden_dim, out_dim, K=k)
        self.relu = tlx.ReLU()
        self.dropout = tlx.layers.Dropout(drop_rate)
        self.name = name
        self.lambda_max = None

    def forward(self, x, edge_index, edge_weight, num_nodes):
        if self.lambda_max is None:
            self.lambda_max = self.__get_max_lambda__(edge_index, edge_weight)
        x = self.conv1(x, edge_index, num_nodes, edge_weight, lambda_max=self.lambda_max)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index, num_nodes, edge_weight, lambda_max=self.lambda_max)
        return x

    def __get_max_lambda__(self, edge_index, edge_weight):
        """Compute largest Laplacian eigenvalue"""
        edge_index = tlx.convert_to_numpy(edge_index)
        row = edge_index[0]
        col = edge_index[1]
        edge_weight = tlx.convert_to_numpy(edge_weight)
        W = csr_matrix((edge_weight, (row, col)))
        L = self.__laplacian__(W)
        return linalg.eigsh(L, k=1, which='LM', return_eigenvectors=False)[0]

    def __laplacian__(self, W, normalized=True):
        """Return graph Laplacian"""
        d = W.sum(axis=0)
        if not normalized:
            D = diags(d.A.squeeze(), 0)
            L = D - W
        else:
            d += np.spacing(np.array(0, W.dtype))
            d = 1 / np.sqrt(d)
            D = diags(d.A.squeeze(), 0)
            I = identity(d.size, dtype=W.dtype)
            L = I - D * W * D

        assert np.abs(L - L.T).mean() < 1e-9
        assert type(L) is csr.csr_matrix
        return L
