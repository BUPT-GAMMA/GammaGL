import functools
import os.path as osp
import shutil

import pytest
from gammagl.data import Dataset


def load_dataset(root: str, name: str, *args, **kwargs) -> Dataset:
    r"""Returns a variety of datasets according to :obj:`name`."""
    if name.lower() in ['cora', 'citeseer', 'pubmed']:
        from gammagl.datasets import Planetoid
        path = osp.join(root, 'Planetoid', name)
        return Planetoid(path, name, *args, **kwargs)
    if name in ['BZR', 'ENZYMES', 'IMDB-BINARY', 'MUTAG']:
        from gammagl.datasets import TUDataset
        path = osp.join(root, 'TUDataset')
        return TUDataset(path, name, *args, **kwargs)
    if name.lower() in ['dblp']:
        from gammagl.datasets import DBLP
        path = osp.join(root, 'DBLP')
        return DBLP(path, *args, **kwargs)
@pytest.fixture(scope='session')
def get_dataset():
    root = osp.join('/', 'tmp', 'ggl_test_datasets')
    yield functools.partial(load_dataset, root)
    if osp.exists(root):
        shutil.rmtree(root)