import numpy as np
import scipy.sparse as sp
import tensorlayerx as tlx
from gammagl.transforms import BaseTransform


class SIGN(BaseTransform):
    r"""The Scalable Inception Graph Neural Network module (SIGN) from the
    `"SIGN: Scalable Inception Graph Neural Networks"
    <https://arxiv.org/abs/2004.11198>`_ paper (functional name: :obj:`sign`),
    which precomputes the fixed representations

    .. math::
        \mathbf{X}^{(i)} = {\left( \mathbf{D}^{-1/2} \mathbf{A}
        \mathbf{D}^{-1/2} \right)}^i \mathbf{X}

    for :math:`i \in \{ 1, \ldots, K \}` and saves them in
    :obj:`data.x1`, :obj:`data.x2`, ...

    .. note::

        Since intermediate node representations are pre-computed, this operator
        is able to scale well to large graphs via classic mini-batching.
        For an example of using SIGN, see `examples/sign.py
        <https://github.com/BUPT-GAMMA/GammaGL/tree/main/examples/sign>`_.

    Parameters
    ----------
    K: int
        The number of hops/layer.
    """
    def __init__(self, K):
        self.K = K

    def __call__(self, graph):
        assert graph.edge_index is not None
        if tlx.is_tensor(graph.edge_index):
            row, col = tlx.convert_to_numpy(graph.edge_index)
        else:
            row, col = graph.edge_index
        weight = np.ones_like(row, dtype=np.float32)

        # Here the graph is undirected.
        deg = np.bincount(row)
        deg_inv_sqrt = np.power(deg, -0.5, dtype=np.float32).flatten()
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        new_weight = deg_inv_sqrt[row] * weight * deg_inv_sqrt[col]
        new_adj = sp.coo_matrix((new_weight, [col, row]))
        assert graph.x is not None
        if tlx.is_tensor(graph.x):
            x = tlx.convert_to_numpy(graph.x)
        else:
            x = graph.x
        xs = [x]
        for i in range(1, self.K + 1):
            xs += [new_adj @ xs[-1]]
            graph[f'x{i}'] = xs[-1]

        return graph

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(K={self.K})'