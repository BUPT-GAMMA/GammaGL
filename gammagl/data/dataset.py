import copy
import json
import os
import os.path as osp
import re
import shutil
import sys
import warnings
from collections.abc import Sequence
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import tensorlayerx as tlx
from tensorlayerx.dataflow import Dataset
from gammagl.data import Graph
from gammagl.data.utils import get_dataset_root, get_dataset_meta_path, md5folder
from gammagl.data.makedirs import makedirs

try:
    import cPickle as pickle
except ImportError:
    import pickle

IndexType = Union[slice, np.ndarray, Sequence]


class Dataset(Dataset):
    r"""Dataset base class for creating graph datasets.
        See `here <https://gammagl.readthedocs.io/en/latest/notes/create_dataset.html#>`__ for the accompanying tutorial.

        Parameters
        ----------
        root: str, optional
            Root directory where the dataset should be
            saved. (optional: :obj:`None`)
        transform: callable, optional
            A function/transform that takes in an
            :obj:`gammagl.data.Graph` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform: callable, optional
            A function/transform that takes in
            an :obj:`gammagl.data.Graph` object and returns a
            transformed version. The graph object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter: callable, optional
            A function that takes in an
            :obj:`gammagl.data.Graph` object and returns a boolean
            value, indicating whether the graph object should be included in the
            final dataset. (default: :obj:`None`)
        force_reload: bool, optional
            Whether to re-process the dataset.(default: :obj:`False`)

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
                 pre_filter: Optional[Callable] = None,
                 force_reload: bool = False):
        super().__init__()

        self.raw_root = root

        assert root is None or isinstance(root, str)
        if root is None:
            root = get_dataset_root()
        else:
            root = osp.abspath(osp.normpath(root))

        self.root = root
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        self._indices: Optional[Sequence] = None
        self.force_reload = force_reload

        # when finished，record dataset path to .ggl/datasets.json
        # next time will use this dataset to avoid download repeatedly.

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

    # for example: PPI at ~/.ggl/datasets/PPI
    @property
    def root_dir(self) -> str:
        if hasattr(self, 'name'):
            return osp.join(self.root, self.name)
        return osp.join(self.root, self._name)

    # for example: PPI at ~/.ggl/datasets/PPI/raw
    @property
    def raw_dir(self) -> str:
        return osp.join(self.root_dir, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root_dir, 'processed')

    @property
    def _name(self) -> str:
        return self.__class__.__name__

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

    def rb_config(self, config_dict, file):
        file.seek(0)
        json.dump(config_dict, file)
        file.truncate()

    def _download(self):
        if hasattr(self, "name"):
            name = self.name
        else:
            name = self._name
        dataset_meta_path = get_dataset_meta_path()
        if files_exist(self.raw_paths) or files_exist(self.processed_paths):
            # for compatibility, check config file
            with open(dataset_meta_path, 'r+') as f:
                dataset_meta_dict = json.load(f)
                if name in dataset_meta_dict and 'root_dir' in dataset_meta_dict[name] and 'hash' in \
                        dataset_meta_dict[name]:
                    return
                # dataset_meta = dict()
                # dataset_meta['root_dir'] = osp.abspath(self.root_dir)
                dataset_meta_dict[name] = {
                    'root_dir': self.root_dir,
                    'hash': md5folder(self.raw_dir)
                }

                self.rb_config(dataset_meta_dict, f)
            return

        with open(dataset_meta_path, 'r+') as f:
            dataset_meta_dict = json.load(f)
            if name in dataset_meta_dict:
                dataset_meta = dataset_meta_dict[name]
                record_root_dir = dataset_meta.get('root_dir', None)
                if record_root_dir is None:
                    del dataset_meta_dict[name]
                    self.rb_config(dataset_meta_dict, f)
                else:
                    # validate
                    record_raw_dir = osp.join(osp.join(record_root_dir, 'raw'))
                    if osp.exists(record_raw_dir) and dataset_meta.get('hash', '') == md5folder(
                            osp.join(record_raw_dir)):
                        if osp.exists(self.root_dir):
                            # shutil.rmtree(self.root_dir)
                            if osp.isfile(self.root_dir):
                                raise FileExistsError(
                                    f"Settled dataset root:{self.root_dir} is existed! Please clear it first.")
                            elif len(os.listdir(self.root_dir)) != 0:
                                files = os.listdir(self.root_dir)
                                if len(files) >= 3:
                                    raise FileExistsError(
                                        f"Settled dataset root:{self.root_dir} is not empty! Please clear it first.")
                                if (len(files) == 1 and 'raw' in files or 'processed' in files) \
                                        or ('raw' in files and 'processed' in files):
                                    # download error before
                                    shutil.rmtree(self.root_dir)
                            else:
                                os.rmdir(self.root_dir)
                        print(
                            f"Dataset[{name}] has been downloaded, now copy it from {record_root_dir} to {self.root_dir}.")
                        shutil.copytree(record_root_dir, self.root_dir)
                        # default position, update it
                        if self.raw_root is None:
                            dataset_meta['root'] = self.root
                            dataset_meta_dict[name] = dataset_meta
                            self.rb_config(dataset_meta_dict, f)
                        # success and return
                        return
                    # else download it

            # download
            makedirs(self.raw_dir)
            self.download()

            # success
            dataset_meta_dict[name] = {
                'root_dir': self.root_dir,
                'hash': md5folder(self.raw_dir)
            }
            self.rb_config(dataset_meta_dict, f)

            # pass
        # self.download()
        # success

    def _process(self):
        f = osp.join(self.processed_dir, tlx.BACKEND + '_pre_transform.pt')
        if osp.exists(f) and self.load_with_pickle(f) != _repr(self.pre_transform):
            warnings.warn(
                f"The `pre_transform` argument differs from the one used in "
                f"the pre-processed version of this dataset. If you want to "
                f"make use of another pre-processing technique, make sure to "
                f"sure to delete '{self.processed_dir}' first"
                f"`force_reload=True` explicitly to reload the dataset.")

        f = osp.join(self.processed_dir, tlx.BACKEND + '_pre_filter.pt')
        if osp.exists(f) and self.load_with_pickle(f) != _repr(self.pre_filter):
            warnings.warn(
                "The `pre_filter` argument differs from the one used in the "
                "pre-processed version of this dataset. If you want to make "
                "use of another pre-fitering technique, make sure to delete "
                "'{self.processed_dir}' first"
                "`force_reload=True` explicitly to reload the dataset.")

        if not self.force_reload and files_exist(self.processed_paths):  # pragma: no cover
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

            Parameters
            ----------
            return_perm: bool, optional
                If set to :obj:`True`, will also
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
