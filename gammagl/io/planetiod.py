import sys
import itertools
import numpy as np
import os.path as osp
from gammagl.data import Graph
from gammagl.io import read_txt_array
from gammagl.utils import remove_self_loops
from gammagl.utils import coalesce as coalesce_fn
from itertools import repeat

try:
    import cPickle as pickle
except ImportError:
    import pickle


def read_planetoid_data(folder, prefix):
    names = ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']
    items = [read_file(folder, prefix, name) for name in names]
    x, tx, allx, y, ty, ally, graph, test_index = items
    train_index = np.arange(y.shape[0])
    val_index = np.arange(y.shape[0], y.shape[0] + 500)
    sorted_test_index = np.sort(test_index)

    if prefix.lower() == 'citeseer':
        # There are some isolated nodes in the Citeseer graph, resulting in
        # none consecutive test indices. We need to identify them and add them
        # as zero vectors to `tx` and `ty`.
        len_test_indices = (test_index.max() - test_index.min()).item() + 1

        tx_ext = np.zeros((len_test_indices, tx.shape[1]))
        tx_ext[sorted_test_index - test_index.min(), :] = tx
        ty_ext = np.zeros((len_test_indices, ty.shape[1]))
        ty_ext[sorted_test_index - test_index.min(), :] = ty

        tx, ty = tx_ext, ty_ext

    x = np.concatenate([allx, tx])
    x[test_index] = x[sorted_test_index]

    y = np.concatenate([ally, ty]).argmax(1)
    y[test_index] = y[sorted_test_index]

    train_mask = index_to_mask(train_index, size=y.shape[0])
    val_mask = index_to_mask(val_index, size=y.shape[0])
    test_mask = index_to_mask(test_index, size=y.shape[0])

    edge_index = edge_index_from_dict(graph, num_nodes=y.shape[0])

    data = Graph(edge_index=edge_index, x=x, y=y)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    data.tensor()
    return data


def read_file(folder, prefix, name):
    path = osp.join(folder, f'ind.{prefix.lower()}.{name}')

    if name == 'test.index':
        return read_txt_array(path)

    with open(path, 'rb') as f:
        if sys.version_info > (3, 0):
            out = pickle.load(f, encoding='latin1')
        else:
            out = pickle.load(f)

    if name == 'graph':
        return out

    out = out.todense() if hasattr(out, 'todense') else out
    out = np.array(out)
    return out


def edge_index_from_dict(graph_dict, num_nodes=None, coalesce=True):
    row, col = [], []
    for key, value in graph_dict.items():
        row += repeat(key, len(value))
        col += value
    edge_index = np.stack([np.array(row), np.array(col)], axis=0)
    if coalesce:
        # NOTE: There are some duplicated edges and self loops in the datasets.
        #       Other implementations do not remove them!
        edge_index, _ = remove_self_loops(edge_index)
        edge_index = coalesce_fn(edge_index)
    return edge_index


def index_to_mask(index, size):
    mask = np.zeros((size, ), dtype=bool)
    mask[index] = 1
    return mask