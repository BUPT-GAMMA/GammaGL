from typing import Callable, List, Union

from gammagl.data import Graph, HeteroGraph
from gammagl.transforms import BaseTransform


class Compose(BaseTransform):
    """Composes several transforms together.

    Parameters
    ----------
    transforms: List[Callable]
        List of transforms to compose.

    """
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, graph: Graph):
        for transform in self.transforms:
            if isinstance(graph, (list, tuple)):
                graph = [transform(d) for d in graph]
            else:
                graph = transform(graph)
        return graph

    def __repr__(self) -> str:
        args = [f'  {transform}' for transform in self.transforms]
        return '{}([\n{}\n])'.format(self.__class__.__name__, ',\n'.join(args))
