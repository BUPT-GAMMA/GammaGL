import copy
import os.path as osp
import re
import sys
import warnings
from collections.abc import Sequence
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import tensorlayerx as tlx
from tensorlayerx.dataflow import Dataset
from gammagl.data import Graph
from gammagl.data.makedirs import makedirs

try:
    import cPickle as pickle
except ImportError:
    import pickle

IndexType = Union[slice, np.ndarray, Sequence]


class Dataset(Dataset):
    r"""Dataset base class for creating graph datasets.
    See `here <https://gammagl.readthedocs.io/en/latest/notes/create_dataset.html#l>`__ for the accompanying tutorial.

    Args:
        root (string, optional): Root directory where the dataset should be
            saved. (optional: :obj:`None`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`gammagl.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`gammagl.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`gammagl.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        r"""The name of the files in the :obj:`self.raw_dir` folder that must
        be present in order to skip downloading."""
        raise NotImplementedError

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        r"""The name of the files in the :obj:`self.processed_dir` folder that
        must be present in order to skip processing."""
        raise NotImplementedError

    def download(self):
        r"""Downloads the dataset to the :obj:`self.raw_dir` folder."""
        raise NotImplementedError

    def process(self):
        r"""Processes the dataset to the :obj:`self.processed_dir` folder."""
        raise NotImplementedError

    def len(self) -> int:
        r"""Returns the number of graphs stored in the dataset."""
        raise NotImplementedError

    def get(self, idx: int) -> Graph:
        r"""Gets the data object at index :obj:`idx`."""
        raise NotImplementedError

    def __init__(self, root: Optional[str] = None,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        super().__init__()

        if isinstance(root, str):
            root = osp.expanduser(osp.normpath(root))

        self.root = root
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        self._indices: Optional[Sequence] = None

        if 'download' in self.__class__.__dict__:
            self._download()

        if 'process' in self.__class__.__dict__:
            self._process()

    def save_with_pickle(self, obj, file_name):
        with open(file_name, 'wb') as f:
            pickle.dump(obj, f)
        return True

    def load_with_pickle(self, file_name):
        with open(file_name, 'rb') as f:
            obj = pickle.load(f)
        return obj

    def save_data(self, obj, file_name):
        r"""Support save data according to different backend."""
        if tlx.BACKEND == 'paddle':
            # with open(file_name, 'wb') as f:
            #     pickle.dump(obj, f)
            import paddle
            obj[0].numpy()
            paddle.save(obj, file_name)
        elif tlx.BACKEND == 'torch':
            import torch
            torch.save(obj, file_name)
        else:
            with open(file_name, 'wb') as f:
                obj[0].numpy()
                pickle.dump(obj, f)
        return True

    def load_data(self, file_name):
        r"""Support load data according to different backend."""
        if tlx.BACKEND == 'paddle':
            # with open(file_name, 'rb') as f:
            #     obj = pickle.load(f)
            #     obj[0].tensor()
            #     if obj[1]:
            #         obj = [obj[0], {item: obj[1][item][1] for item in obj[1]}]
            # with open(file_name, 'rb') as f:
            #     obj = pickle.load(f)
            import paddle
            obj = paddle.load(file_name, return_numpy=True)
            obj[0].tensor()
        elif tlx.BACKEND == 'torch':
            import torch
            id = torch.tensor(1).get_device()
            if id != -1:
                device = 'cuda:' + str(id)
            else:
                device = 'cpu'
            obj = torch.load(file_name, map_location=device)
        else:
            with open(file_name, 'rb') as f:
                obj = pickle.load(f)
                obj[0].tensor()
        return obj

    def indices(self) -> Sequence:
        return range(self.len()) if self._indices is None else self._indices

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'processed')

    @property
    def num_node_features(self) -> int:
        r"""Returns the number of features per node in the dataset."""
        data = self[0]
        data = data[0] if isinstance(data, tuple) else data
        if hasattr(data, 'num_node_features'):
            return data.num_node_features
        raise AttributeError(f"'{data.__class__.__name__}' object has no "
                             f"attribute 'num_node_features'")

    @property
    def num_features(self) -> int:
        r"""Returns the number of features per node in the dataset.
        Alias for :py:attr:`~num_node_features`."""
        return self.num_node_features

    @property
    def num_edge_features(self) -> int:
        r"""Returns the number of features per edge in the dataset."""
        data = self[0]
        data = data[0] if isinstance(data, tuple) else data
        if hasattr(data, 'num_edge_features'):
            return data.num_edge_features
        raise AttributeError(f"'{data.__class__.__name__}' object has no "
                             f"attribute 'num_edge_features'")

    @property
    def raw_paths(self) -> List[str]:
        r"""The absolute filepaths that must be present in order to skip
        downloading."""
        files = to_list(self.raw_file_names)
        return [osp.join(self.raw_dir, f) for f in files]

    @property
    def processed_paths(self) -> List[str]:
        r"""The absolute filepaths that must be present in order to skip
        processing."""
        files = to_list(self.processed_file_names)
        return [osp.join(self.processed_dir, f) for f in files]

    def _download(self):
        if files_exist(self.raw_paths):  # pragma: no cover
            return

        makedirs(self.raw_dir)
        self.download()

    def _process(self):
        f = osp.join(self.processed_dir, tlx.BACKEND + '_pre_transform.pt')
        if osp.exists(f) and self.load_with_pickle(f) != _repr(self.pre_transform):
            warnings.warn(
                f"The `pre_transform` argument differs from the one used in "
                f"the pre-processed version of this dataset. If you want to "
                f"make use of another pre-processing technique, make sure to "
                f"sure to delete '{self.processed_dir}' first")

        f = osp.join(self.processed_dir, tlx.BACKEND + '_pre_filter.pt')
        if osp.exists(f) and self.load_with_pickle(f) != _repr(self.pre_filter):
            warnings.warn(
                "The `pre_filter` argument differs from the one used in the "
                "pre-processed version of this dataset. If you want to make "
                "use of another pre-fitering technique, make sure to delete "
                "'{self.processed_dir}' first")

        if files_exist(self.processed_paths):  # pragma: no cover
            # self.process()
            return

        print('Processing...', file=sys.stderr)

        makedirs(self.processed_dir)
        self.process()

        path = osp.join(self.processed_dir, tlx.BACKEND + '_pre_transform.pt')
        self.save_with_pickle(_repr(self.pre_transform), path)
        path = osp.join(self.processed_dir, tlx.BACKEND + '_pre_filter.pt')
        self.save_with_pickle(_repr(self.pre_filter), path)

        print('Done!', file=sys.stderr)

    def __len__(self) -> int:
        r"""The number of examples in the dataset."""
        return len(self.indices())

    def __getitem__(
            self,
            idx: Union[int, np.integer, IndexType],
    ) -> Union['Dataset', Graph]:
        r"""In case :obj:`idx` is of type integer, will return the data object
        at index :obj:`idx` (and transforms it in case :obj:`transform` is
        present).
        In case :obj:`idx` is a slicing object, *e.g.*, :obj:`[2:5]`, a list, a
        tuple, or a :obj:`Tensor` or :obj:`np.ndarray` of type long or
        bool, will return a subset of the dataset at the specified indices."""
        if (isinstance(idx, (int, np.integer))
                or (isinstance(idx, np.ndarray) and np.isscalar(idx))):

            data = self.get(self.indices()[idx])
            data = data if self.transform is None else self.transform(data)
            return data

        else:
            return self.index_select(idx)

    def index_select(self, idx: IndexType) -> 'Dataset':
        r"""Creates a subset of the dataset from specified indices :obj:`idx`.
        Indices :obj:`idx` can be a slicing object, *e.g.*, :obj:`[2:5]`, a
        list, a tuple, or a :obj:`Tensor` or :obj:`np.ndarray` of type
        long or bool."""
        indices = self.indices()

        if isinstance(idx, slice):
            indices = indices[idx]

        elif tlx.ops.is_tensor(idx) and idx.dtype == tlx.long:
            return self.index_select(idx.flatten().tolist())

        elif tlx.ops.is_tensor(idx) and idx.dtype == tlx.bool:
            idx = idx.flatten().nonzero(as_tuple=False)
            return self.index_select(idx.flatten().tolist())

        elif isinstance(idx, np.ndarray) and idx.dtype == np.int64:
            return self.index_select(idx.flatten().tolist())

        elif isinstance(idx, np.ndarray) and idx.dtype == np.bool:
            idx = idx.flatten().nonzero()[0]
            return self.index_select(idx.flatten().tolist())

        elif isinstance(idx, Sequence) and not isinstance(idx, str):
            indices = [indices[i] for i in idx]

        else:
            raise IndexError(
                f"Only slices (':'), list, tuples, Tensor and "
                f"np.ndarray of dtype long or bool are valid indices (got "
                f"'{type(idx).__name__}')")

        dataset = copy.copy(self)
        dataset._indices = indices
        return dataset

    def shuffle(
            self,
            return_perm: bool = False,
    ):
        #    -> Union['Dataset', Tuple['Dataset', tf.Tensor]]:
        r"""Randomly shuffles the examples in the dataset.

        Args:
            return_perm (bool, optional): If set to :obj:`True`, will also
                return the random permutation used to shuffle the dataset.
                (default: :obj:`False`)
        """
        perm = np.random.permutation(len(self))
        dataset = self.index_select(perm)
        return (dataset, perm) if return_perm is True else dataset

    def __repr__(self) -> str:
        arg_repr = str(len(self)) if len(self) > 1 else ''
        return f'{self.__class__.__name__}({arg_repr})'


def to_list(value: Any) -> Sequence:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    else:
        return [value]


def files_exist(files: List[str]) -> bool:
    # NOTE: We return `False` in case `files` is empty, leading to a
    # re-processing of files on every instantiation.
    return len(files) != 0 and all([osp.exists(f) for f in files])


def _repr(obj: Any) -> str:
    if obj is None:
        return 'None'
    return re.sub('(<.*?)\\s.*(>)', r'\1\2', obj.__repr__())
