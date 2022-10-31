import copy
from collections.abc import Mapping,Sequence
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

from gammagl.data import Graph
from gammagl.data.collate import collate
from gammagl.data.dataset import Dataset, IndexType
from gammagl.data.separate import separate
import tensorlayerx as tlx


class InMemoryDataset(Dataset):
    r"""Dataset base class for creating graph datasets which easily fit
    into CPU memory.
    Inherits from :class:`gammagl.data.Dataset`.
    See `here <https://gammagl.readthedocs.io/en/latest/notes/create_dataset.html#creating-in-memory-datasets>`__ for the accompanying tutorial.

    Args:
        root (string, optional): Root directory where the dataset should be
            saved. (default: :obj:`None`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`gammagl.data.Graph` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`gammagl.data.Graph` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`gammagl.data.Graph` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """
    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        raise NotImplementedError

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        raise NotImplementedError

    def download(self):
        raise NotImplementedError

    def process(self):
        raise NotImplementedError

    def __init__(self, root: Optional[str] = None,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data = None
        self.slices = None
        self._data_list: Optional[List[Graph]] = None

    @property
    def num_classes(self) -> int:
        r"""Returns the number of classes in the dataset."""
        y = self.data.y
        if y is None:
            return 0
        # elif y.numel() == y.size(0) and not torch.is_floating_point(y):
        #     return int(self.data.y.max()) + 1
        elif y.ndim == 1:
            y = tlx.convert_to_numpy(y)
            return int(y.max() + 1)
        else:
            return self.data.y.shape[-1]

    def len(self) -> int:
        if self.slices is None:
            return 1
        for _, value in nested_iter(self.slices):
            return len(value) - 1
        return 0

    def get(self, idx: int) -> Graph:
        if self.len() == 1:
            return copy.copy(self.data)

        if not hasattr(self, '_data_list') or self._data_list is None:
            self._data_list = self.len() * [None]
        elif self._data_list[idx] is not None:
            return copy.copy(self._data_list[idx])

        data = separate(
            cls=self.data.__class__,
            batch=self.data,
            idx=idx,
            slice_dict=self.slices,
            decrement=False,
        )
        
        self._data_list[idx] = copy.copy(data)
        return data

    @staticmethod
    def collate(
            data_list: List[Graph]):
            #-> Tuple[Graph, Optional[Dict[str, Tensor]]]:
        r"""Collates a Python list of :obj:`gammagl.data.Data` objects
        to the internal storage format of
        :class:`~gammagl.data.InMemoryDataset`."""
        if len(data_list) == 1:
            return data_list[0], None

        data, slices, _ = collate(
            data_list[0].__class__,
            data_list=data_list,
            increment=False,
            add_batch=False,
        )

        return data, slices

    def copy(self, idx: Optional[IndexType] = None) -> 'InMemoryDataset':
        r"""Performs a deep-copy of the dataset. If :obj:`idx` is not given,
        will clone the full dataset. Otherwise, will only clone a subset of the
        dataset from indices :obj:`idx`.
        Indices can be slices, lists, tuples, and a :obj:`Tensor` or
        :obj:`np.ndarray` of type long or bool.
        """
        if idx is None:
            data_list = [self.get(i) for i in range(len(self))]
        else:
            data_list = [self.get(i) for i in self.index_select(idx).indices()]

        dataset = copy.copy(self)
        dataset._indices = None
        dataset._data_list = data_list
        dataset.data, dataset.slices = self.collate(data_list)
        return dataset


def nested_iter(node):
    if isinstance(node, Mapping):
        for key, value in node.items():
            for inner_key, inner_value in nested_iter(value):
                yield inner_key, inner_value
    elif isinstance(node, Sequence):
        for i, inner_value in enumerate(node):
            yield i, inner_value
    else:
        yield None, node
