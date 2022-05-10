from gammagl.data import Graph


class BaseTransform:
	r"""An abstract base class for writing transforms.
	Transforms are a general way to modify and customize
	:class:`~torch_geometric.data.Data` objects, either by implicitly passing
	them as an argument to a :class:`~torch_geometric.data.Dataset`, or by
	applying them explicitly to individual :class:`~torch_geometric.data.Data`
	objects.
	.. code-block:: python
		import torch_geometric.transforms as T
		from torch_geometric.datasets import TUDataset
		transform = T.Compose([T.ToUndirected(), T.AddSelfLoops()])
		dataset = TUDataset(path, name='MUTAG', transform=transform)
		data = dataset[0]  # Implicitly transform data on every access.
		data = TUDataset(path, name='MUTAG')[0]
		data = transform(data)  # Explicitly transform data.
	"""
	def __call__(self, data):
		raise NotImplementedError
	
	def __repr__(self) -> str:
		return f'{self.__class__.__name__}()'



from typing import List
import numpy as np
import tensorlayerx as tlx
class NormalizeFeatures(BaseTransform):
    r"""Row-normalizes the attributes given in :obj:`attrs` to sum-up to one
    (functional name: :obj:`normalize_features`).

    Args:
        attrs (List[str]): The names of attributes to normalize.
            (default: :obj:`["x"]`)
    """
    def __init__(self, attrs: List[str] = ["x"]):
        self.attrs = attrs

    def __call__(self, graph: Graph):
        for store in graph.stores:
            for key, value in store.items(*self.attrs):
                if not isinstance(value, np.ndarray):
                    value = value.numpy()
                value = value - value.min()
                value = np.divide(value, value.sum(axis=-1, keepdims=True).clip(min=1.))
                store[key] = tlx.convert_to_tensor(value, dtype=tlx.float32)
        return graph

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
