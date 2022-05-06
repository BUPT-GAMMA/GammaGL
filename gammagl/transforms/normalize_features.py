from typing import List, Union

from gammagl.data import BaseGraph, HeteroGraph
from gammagl.transforms import BaseTransform
import numpy as np


class NormalizeFeatures(BaseTransform):
    r"""Row-normalizes the attributes given in :obj:`attrs` to sum-up to one.

    Args:
        attrs (List[str]): The names of attributes to normalize.
            (default: :obj:`["x"]`)
    """
    def __init__(self, attrs: List[str] = ["x"]):
        self.attrs = attrs

    def __call__(self, data: Union[BaseGraph, HeteroGraph]):
        for store in data.stores:
            for key, value in store.items(*self.attrs):
                value = np.divide(value, value.sum(axis=-1, keepdims=True).clip(min=1.))
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'