import inspect
from collections.abc import Sequence
from typing import Any, List, Optional, Union

import numpy as np
import tensorlayerx as tlx
from gammagl.data import Graph
from gammagl.data.collate import collate
from gammagl.data.dataset import IndexType
from gammagl.data.separate import separate


class DynamicInheritance(type):
    # A meta class that sets the base class of a `Batch` object, e.g.:
    # * `Batch(Graph)` in case `Graph` objects are batched together
    # * `Batch(HeteroGraph)` in case `HeteroGraph` objects are batched together
    def __call__(cls, *args, **kwargs):
        base_cls = kwargs.pop('_base_cls', Graph)

        if issubclass(base_cls, BatchGraph):
            new_cls = base_cls
        else:
            name = f'{base_cls.__name__}{cls.__name__}'
            if name not in globals():
                globals()[name] = type(name, (cls, base_cls), {})
            new_cls = globals()[name]
        # https://docs.python.org/zh-cn/3.7/library/inspect.html#introspecting-callables-with-the-signature-object
        params = list(inspect.signature(base_cls.__init__).parameters.items())
        for i, (k, v) in enumerate(params[1:]):
            if k == 'args' or k == 'kwargs':
                continue
            if i < len(args) or k in kwargs:
                continue
            if v.default is not inspect.Parameter.empty:
                continue
            kwargs[k] = None

        return super(DynamicInheritance, new_cls).__call__(*args, **kwargs)


class DynamicInheritanceGetter(object):
    def __call__(self, cls, base_cls):
        return cls(_base_cls=base_cls)


class BatchGraph(metaclass=DynamicInheritance):
    r"""A data object describing a batch of graphs as one big (disconnected)
    graph.
    Inherits from :class:`gammagl.data.Graph` or
    :class:`gammagl.data.HeteroGraph`.
    In addition, single graphs can be identified via the assignment vector
    :obj:`batch`, which maps each node to its respective graph identifier.
    """
    @classmethod
    def from_data_list(cls, data_list: List[Graph],
                       follow_batch: Optional[List[str]] = None,
                       exclude_keys: Optional[List[str]] = None):
        # https://github.com/pyg-team/pytorch_geometric/issues/3332
        r"""Constructs a :class:`~gammagl.data.BatchGraph` object from a
        Python list of :class:`~gammagl.data.Graph` or
        :class:`~gammagl.data.HeteroGraph` objects.
        The assignment vector :obj:`batch` is created on the fly.
        In addition, creates assignment vectors for each key in
        :obj:`follow_batch`.
        Will exclude any keys given in :obj:`exclude_keys`."""

        batch, slice_dict, inc_dict = collate(
            cls,
            data_list=data_list,
            increment=True,
            add_batch=not isinstance(data_list[0], BatchGraph),
            follow_batch=follow_batch,
            exclude_keys=exclude_keys,
        )

        batch._num_graphs = len(data_list)
        batch._slice_dict = slice_dict
        batch._inc_dict = inc_dict

        return batch

    def get_example(self, idx: int) -> Graph:
        r"""Gets the :class:`~gammagl.data.Graph` or
        :class:`~gammagl.data.HeteroGraph` object at index :obj:`idx`.
        The :class:`~gammagl.data.BatchGraph` object must have been created
        via :meth:`from_data_list` in order to be able to reconstruct the
        initial object."""

        if not hasattr(self, '_slice_dict'):
            raise RuntimeError(
                ("Cannot reconstruct 'Data' object from 'Batch' because "
                 "'Batch' was not created via 'Batch.from_data_list()'"))

        data = separate(
            cls=self.__class__.__bases__[-1],
            batch=self,
            idx=idx,
            slice_dict=self._slice_dict,
            inc_dict=self._inc_dict,
            decrement=True,
        )

        return data

    def index_select(self,
                     idx: IndexType) -> List[Graph]:
        r"""Creates a subset of :class:`~gammagl.data.Graph` or
        :class:`~gammagl.data.HeteroGraph` objects from specified
        indices :obj:`idx`.
        Indices :obj:`idx` can be a slicing object, *e.g.*, :obj:`[2:5]`, a
        list, a tuple, or a :obj:`torch.Tensor` or :obj:`np.ndarray` of type
        long or bool.
        The :class:`~gammagl.data.BatchGraph` object must have been created
        via :meth:`from_data_list` in order to be able to reconstruct the
        initial objects."""
        if isinstance(idx, slice):
            idx = list(range(self.num_graphs)[idx])

        elif tlx.is_tensor(idx) and idx.dtype == tlx.int64:
            idx = tlx.convert_to_numpy(idx).flatten().tolist()

        elif tlx.is_tensor(idx) and idx.dtype == tlx.bool:
            idx = tlx.convert_to_numpy(idx).flatten().nonzero()[0].flatten().tolist()

        elif isinstance(idx, np.ndarray) and idx.dtype == np.int64:
            idx = idx.flatten().tolist()

        elif isinstance(idx, np.ndarray) and idx.dtype == bool:
            idx = idx.flatten().nonzero()[0].flatten().tolist()

        elif isinstance(idx, Sequence) and not isinstance(idx, str):
            pass

        else:
            raise IndexError(
                f"Only slices (':'), list, tuples, torch.tensor and "
                f"np.ndarray of dtype long or bool are valid indices (got "
                f"'{type(idx).__name__}')")

        return [self.get_example(i) for i in idx]

    def __getitem__(self, idx: Union[int, np.integer, str, IndexType]) -> Any:
        if (isinstance(idx, (int, np.integer))
                or (tlx.is_tensor(idx) and idx.ndim() == 0)
                or (isinstance(idx, np.ndarray) and np.isscalar(idx))):
            return self.get_example(idx)
        elif isinstance(idx, str) or (isinstance(idx, tuple)
                                      and isinstance(idx[0], str)):
            # Accessing attributes or node/edge types:
            return super().__getitem__(idx)
        else:
            return self.index_select(idx)

    def to_data_list(self) -> List[Graph]:
        r"""Reconstructs the list of :class:`~gammagl.data.Graph` or
        :class:`~gammagl.data.HeteroGraph` objects from the
        :class:`~gammagl.data.BatchGraph` object.
        The :class:`~gammagl.data.BatchGraph` object must have been created
        via :meth:`from_data_list` in order to be able to reconstruct the
        initial objects."""
        return [self.get_example(i) for i in range(self.num_graphs)]

    @property
    def num_graphs(self) -> int:
        """Returns the number of graphs in the batch."""
        if hasattr(self, '_num_graphs'):
            return self._num_graphs
        elif hasattr(self, 'ptr'):
            return self.ptr.numel() - 1
        elif hasattr(self, 'batch'):
            return int(self.batch.max()) + 1
        else:
            raise ValueError("Can not infer the number of graphs")

    def __reduce__(self):
        state = self.__dict__.copy()
        return DynamicInheritanceGetter(), self.__class__.__bases__, state