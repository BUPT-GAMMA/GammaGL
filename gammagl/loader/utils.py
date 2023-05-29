# -*- coding: utf-8 -*-
# @author WuJing
# @created 2023/3/2

import copy
import random
import numpy as np
import tensorlayerx as tlx
from typing import Union, Dict, Set, Optional, Any, Tuple, List

import gammagl
from gammagl.data import Graph, HeteroGraph
from gammagl.data.storage import EdgeStorage, NodeStorage
from gammagl.ops.sparse import ind2ptr
from gammagl.utils.platform_utils import all_to_numpy


class DataLoaderIter:
    r"""A data loader iterator extended by a simple post transformation
    function :meth:`transform_fn`. While the iterator may request items from
    different sub-processes, :meth:`transform_fn` will always be executed in
    the main process.

    This iterator is used in PyG's sampler classes, and is responsible for
    feature fetching and filtering data objects after sampling has taken place
    in a sub-process. This has the following advantages:

    * We do not need to share feature matrices across processes which may
      prevent any errors due to too many open file handles.
    * We can execute any expensive post-processing commands on the main thread
      with full parallelization power (which usually executes faster).
    * It lets us naturally support data already being present on the GPU.
    """

    def __init__(self, iter, transform_fn):
        self.iterator = iter
        self.transform_fn = transform_fn

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.iterator)

    def __next__(self):
        return self.transform_fn(next(self.iterator))


def filter_graph(graph, node, row, col, edge, perm):
    out = copy.copy(graph)
    node_process(graph.node_stores, out.node_stores, node)
    edge_process(graph.edge_stores, out.edge_stores, row, col, edge, perm)
    return out


def filter_hetero_graph(graph: HeteroGraph, node_dict: Dict, row_dict: Dict, col_dict: Dict, edge_dict: Dict,
                        perm_dict: Dict):
    out = copy.copy(graph)
    for node_type in graph.node_types:
        node_process(graph[node_type], out[node_type], node_dict[node_type])

    for edge_type in graph.edge_types:
        edge_process(graph[edge_type], out[edge_type], row_dict[edge_type], col_dict[edge_type], edge_dict[edge_type],
                     perm_dict[edge_type] if perm_dict else None)
    return out


def node_process(src_graph: Union[NodeStorage, List[NodeStorage]], tar_graph, index):
    if isinstance(src_graph, (List,)):
        src_graph = src_graph[0]
    if isinstance(tar_graph, (List,)):
        tar_graph = tar_graph[0]

    if src_graph.num_nodes is not None:
        tar_graph.num_nodes = tlx.convert_to_tensor(index.shape[0], dtype=index.dtype)

    for key, value in src_graph.items():
        if src_graph.is_node_attr(key):
            dim = src_graph._parent().__cat_dim__(key, value, src_graph)
            tar_graph[key] = tlx.gather(value, index, axis=dim)
    return src_graph


def edge_process(src_graph: Union[EdgeStorage, List[EdgeStorage]], tar_graph, row, col, index, perm):
    if isinstance(src_graph, (List,)):
        src_graph = src_graph[0]
    if isinstance(tar_graph, (List,)):
        tar_graph = tar_graph[0]
    for key, value in src_graph.items():
        if key == 'edge_index':
            edge_index = tlx.stack([row, col], axis=0)
            tar_graph.edge_index = edge_index
        elif src_graph.is_edge_attr(key):
            dim = src_graph._parent().__cat_dim__(key, value, src_graph)
            if perm is None:
                tar_graph[key] = tlx.gather(value, index, axis=dim)
            else:
                tar_graph[key] = tlx.gather(value, perm[index], axis=dim)
    return src_graph


def get_input_nodes_index(graph: Union[Graph, HeteroGraph], input_nodes=None):
    def to_index(index):
        if index.dtype == tlx.bool:
            return tlx.convert_to_tensor(np.reshape(np.nonzero(tlx.convert_to_numpy(index)), -1))
        return index

    if isinstance(graph, Graph):
        if input_nodes is None:
            return None, range(graph.num_nodes)
        return None, to_index(input_nodes)

    if isinstance(input_nodes, str):
        return input_nodes, range(graph[input_nodes].num_nodes)
    assert isinstance(input_nodes, (list, tuple))
    assert len(input_nodes) == 2
    assert isinstance(input_nodes[0], str)

    node_type, input_nodes = input_nodes
    if input_nodes is None:
        return node_type, range(graph[node_type].num_nodes)
    return node_type, to_index(input_nodes)


# csc format
def to_csc(graph: Union[Graph, EdgeStorage], device, is_sorted):
    perm = None
    # for dense graph
    if graph.edge_index is not None:
        (row, col) = graph.edge_index
        if not is_sorted:
            perm = tlx.argsort(tlx.add((col * graph.size(0)), row))
            row = tlx.gather(row, perm)

        colptr = gammagl.ops.sparse.ind2ptr(tlx.gather(col, perm), graph.size(1))

    else:
        row = tlx.zeros(0)
        colptr = tlx.zeros(graph.num_nodes + 1, dtype=tlx.int64, device=device)

    return colptr, row, perm


# csr format
def to_csr(graph: Union[Graph, EdgeStorage], device, is_sorted):
    perm = None
    # for dense graph
    if graph.edge_index is not None:
        (row, col) = graph.edge_index
        if not is_sorted:
            perm = tlx.argsort(tlx.add((row * graph.size(1)), col))
            col = tlx.gather(col, perm)

        rowptr = gammagl.ops.sparse.ind2ptr(tlx.convert_to_numpy(tlx.gather(row, perm)), graph.size(0))

    else:
        col = tlx.zeros(0)
        rowptr = tlx.zeros(graph.num_nodes + 1, dtype=tlx.int64, device=device)

    return rowptr, col, perm


def to_hetero_csc(graph: HeteroGraph, device=None, is_sorted=False):
    colptr_dict, row_dict, perm_dict = {}, {}, {}
    for store in graph.edge_stores:
        key = store._key
        out = to_csc(store, device, is_sorted)
        colptr_dict[key], row_dict[key], perm_dict[key] = out

    return colptr_dict, row_dict, perm_dict


def remap_keys(original: Dict, mapping: Dict, exclude: Optional[Set[Any]] = None):
    exclude = exclude or set()
    return {
        (k if k in exclude else mapping[k]): v
        for k, v in original.items()
    }

