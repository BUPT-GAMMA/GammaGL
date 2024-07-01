from gammagl.transforms import BaseTransform

from gammagl.data import Graph
from typing import List
import numpy as np
import tensorlayerx as tlx


class SVDFeatureReduction(BaseTransform):
    r"""Dimensionality reduction of node features via Singular Value Decomposition (SVD)
    (functional name: :obj:`normalize_features`).

    Parameters
    ----------
    out_channels: int
        The dimensionlity of node features after reduction.

    """

    def __init__(self, out_channels: int):
        self.out_channels = out_channels

    def __call__(self, graph: Graph):
        assert graph.x is not None

        if graph.x.shape[-1] > self.out_channels:
            x = tlx.convert_to_numpy(graph.x)
            U, S, _ = np.linalg.svd(x, full_matrices=False)
            U_reduced = U[:, :self.out_channels]
            S_reduced = np.diag(S[:self.out_channels])
            x = np.dot(U_reduced, S_reduced)
            graph.x = tlx.convert_to_tensor(x)

        return graph

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
