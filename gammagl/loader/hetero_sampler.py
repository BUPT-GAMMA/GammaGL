from collections.abc import Sequence
from time import time
import copy
from typing import Any, Callable, Dict, Iterator, List, Optional, Union

from gammagl import sample

from gammagl.data import HeteroGraph
from gammagl.data.storage import EdgeStorage, NodeStorage
import tensorlayerx
import numpy as np
from gammagl.typing import EdgeType


NumNeighbors = Union[List[int], Dict[EdgeType, List[int]]]

class NeighborSampler:
    def __init__(
            self,
            data: HeteroGraph,
            num_neighbors: NumNeighbors,
            replace: bool = False,
            directed: bool = True,
            input_node_type: Optional[str] = None,
            share_memory: bool = False,
    ):
        self.data_cls = data.__class__
        self.num_neighbors = num_neighbors
        self.replace = replace
        self.directed = directed

        if isinstance(data, HeteroGraph):
            # Convert the graph data into a suitable format for sampling.
            # NOTE: Since C++ cannot take dictionaries with tuples as key as
            # input, edge type triplets are converted into single strings.
            out = to_hetero_csc(data)
            self.colptr_dict, self.row_dict = out

            self.node_types, self.edge_types = data.metadata()
            # pyg 这里是可以处理tuple的，我暂时没读到这里，我就先处理list吧
            if isinstance(num_neighbors, list):
                num_neighbors = {key: num_neighbors for key in self.edge_types}
            assert isinstance(num_neighbors, dict)
            self.num_neighbors = {
                edge_type_to_str(key): value
                for key, value in num_neighbors.items()
            }

            self.num_hops = max([len(v) for v in self.num_neighbors.values()])

            assert isinstance(input_node_type, str)
            self.input_node_type = input_node_type

        else:
            raise TypeError(f'NeighborLoader found invalid type: {type(data)}')

    def __call__(self, index):
        index = np.array(index)

        if issubclass(self.data_cls, HeteroGraph):
            node_dic, row_dic, col_dic, edge_dic = sample.hetero_sample(self.node_types, self.edge_types,
                                                                     self.row_dict, self.colptr_dict,
                                                                     {self.input_node_type: index},
                                                                     self.num_neighbors,
                                                                     self.num_hops, self.num_hops,
                                                                     self.directed)
            return node_dic, row_dic, col_dic, edge_dic, np.size(index)


class Hetero_Neighbor_Sampler(tensorlayerx.dataflow.DataLoader):

    def __init__(
            self,
            data: HeteroGraph,
            num_neighbors: NumNeighbors,
            input_nodes,
            replace: bool = False,
            directed: bool = True,
            transform: Callable = None,
            neighbor_sampler: Optional[NeighborSampler] = None,
            **kwargs,
    ):
        if 'dataset' in kwargs:
            del kwargs['dataset']
        if 'collate_fn' in kwargs:
            del kwargs['collate_fn']

        self.data = data

        # Save for PyTorch Lightning < 1.6:
        self.num_neighbors = num_neighbors
        self.input_nodes = input_nodes
        self.replace = replace
        self.directed = directed
        self.transform = transform
        self.neighbor_sampler = neighbor_sampler

        if neighbor_sampler is None:
            input_node_type = get_input_node_type(input_nodes)
            # 1
            self.neighbor_sampler = NeighborSampler(
                data, num_neighbors, replace, directed, input_node_type,
                share_memory=kwargs.get('num_workers', 0) > 0)

        return super().__init__(get_input_node_indices(data, input_nodes),
                                collate_fn=self.neighbor_sampler, **kwargs)

    def transform_fn(self, out: Any):
        # 3
        if isinstance(self.data, HeteroGraph):
            node_dict, row_dict, col_dict, edge_dict, batch_size = out
            data = filter_hetero_data(self.data, node_dict, row_dict, col_dict)
            data[self.neighbor_sampler.input_node_type].batch_size = batch_size
        else:
            raise TypeError("this loader only support HeteroGraph class")

        return data if self.transform is None else self.transform(data)

    def _get_iterator(self) -> Iterator:
        # 2
        return DataLoaderIterator(super()._get_iterator(), self.transform_fn)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


###############################################################################


def get_input_node_type(input_nodes):
    if isinstance(input_nodes, tuple):
        assert isinstance(input_nodes[0], str)
        return input_nodes[0]
    else:
        raise TypeError("input_nodes must be like (str, np.ndarray)")
    return None


def get_input_node_indices(data: HeteroGraph,
                           input_nodes):
    if isinstance(data, HeteroGraph):
        assert isinstance(input_nodes, tuple)
        assert len(input_nodes) == 2
        assert isinstance(input_nodes[0], str)
        if input_nodes[1] is None:
            return range(data[input_nodes[0]].num_nodes)
        input_nodes = input_nodes[1]

    if isinstance(input_nodes, np.ndarray):
        if input_nodes.dtype == np.bool8:
            input_nodes = np.nonzero(input_nodes)[0]
        input_nodes = input_nodes.tolist()
    assert isinstance(input_nodes, Sequence)
    return input_nodes



def index_select(value, index):
    # out = None
    # return torch.index_select(value, 0, index, out=out)
    return value[index]


def edge_type_to_str(edge_type: Union[EdgeType, str]) -> str:
    # Since C++ cannot take dictionaries with tuples as key as input, edge type
    # triplets need to be converted into single strings.
    return edge_type if isinstance(edge_type, str) else '__'.join(edge_type)


def to_hetero_csc(hetero_graph):
    # Convert the heterogeneous graph data into a suitable format for sampling
    # (CSC format).
    # Returns dictionaries holding `colptr` and `row` indices as well as edge
    # permutations for each edge type, respectively.
    # Since C++ cannot take dictionaries with tuples as key as input, edge type
    # triplets are converted into single strings.
    colptr_dict, row_dict, perm_dict = {}, {}, {}

    for store in hetero_graph.edge_stores:
        key = edge_type_to_str(store._key)

        if hasattr(store, 'edge_index'):
            (row, col) = store.edge_index
            indptr = np.concatenate(([0], np.bincount(col).cumsum()))
            ind = np.argsort(col, axis=0)
            row = row[ind]
            colptr_dict[key], row_dict[key] = indptr, row
        else:
            raise AttributeError("HeteroGraph object does not contain attributes "
                                 "'edge_index'")
    return colptr_dict, row_dict


def filter_node_store_(store: NodeStorage, out_store: NodeStorage,
                       index):
    # Filters a node storage object to only hold the nodes in `index`:
    for key, value in store.items():
        if key == 'num_nodes':
            out_store.num_nodes = index.numel()
        elif store.is_node_attr(key):
            out_store[key] = index_select(value, index, dim=0)
    return store


def filter_edge_store_(store: EdgeStorage, out_store: EdgeStorage, row,
                       col):
    # Filters a edge storage object to only hold the edges in `index`,
    # which represents the new graph as denoted by `(row, col)`:
    for key, value in store.items():
        if key == 'edge_index':
            out_store.edge_index = np.stack([row, col], axis=0)
        else:
            raise AttributeError("Now we can only process 'edge_index', we will update soon")
    return store


def filter_hetero_data(heterograph, node_dict, row_dict, col_dict):
    # Filters a heterogeneous data object to only hold nodes in `node` and
    # edges in `edge` for each node and edge type, respectively:
    out = copy.copy(heterograph)

    for node_type in heterograph.node_types:
        filter_node_store_(heterograph[node_type], out[node_type],
                           node_dict[node_type])

    for edge_type in heterograph.edge_types:
        edge_type_str = edge_type_to_str(edge_type)
        filter_edge_store_(heterograph[edge_type], out[edge_type],
                           row_dict[edge_type_str], col_dict[edge_type_str])

    return out

from tensorlayerx.dataflow.utils import _BaseDataLoaderIter
class DataLoaderIterator(object):
    def __init__(self, iterator: _BaseDataLoaderIter, transform_fn: Callable):
        self.iterator = iterator
        self.transform_fn = transform_fn

    def __iter__(self) -> 'DataLoaderIterator':
        return self

    def _reset(self, loader: Any, first_iter: bool = False):
        self.iterator._reset(loader, first_iter)

    def __len__(self) -> int:
        return len(self.iterator)

    def __next__(self) -> Any:
        return self.transform_fn(next(self.iterator))
