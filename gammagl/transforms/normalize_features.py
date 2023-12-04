from gammagl.transforms import BaseTransform

from gammagl.data import Graph
from typing import List
import numpy as np
import tensorlayerx as tlx


class NormalizeFeatures(BaseTransform):
    r"""Row-normalizes the attributes given in :obj:`attrs` to sum-up to one
    (functional name: :obj:`normalize_features`). Compute with Numpy and convert results to Tensor at last.

    Parameters
    ----------
    attrs: List[str]
        The names of attributes to normalize.
        (default: :obj:`["x"]`)

    """

    def __init__(self, attrs: List[str] = ["x"]):
        self.attrs = attrs

    def __call__(self, graph: Graph):
        for store in graph.stores:
            for key, value in store.items(*self.attrs):
                if not isinstance(value, np.ndarray):
                    value = tlx.convert_to_numpy(value)
                value = value - value.min()
                value = np.divide(value, value.sum(axis=-1, keepdims=True).clip(min=1.))
                store[key] = tlx.convert_to_tensor(value, dtype=tlx.float32)
        return graph

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
