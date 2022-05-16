from copy import copy
from .check import check_is_numpy
import tensorlayerx as tlx


def maybe_num_nodes(edge_index, num_nodes=None):
    if num_nodes is not None:
        return num_nodes
    elif tlx.is_tensor(edge_index):
        return int(tlx.reduce_max(edge_index)) + 1 if edge_index is not None else 0 # BUG: mindspore reduce max only support float tensor.
    elif check_is_numpy(edge_index):
        return edge_index.max() + 1
    else:
        raise ValueError('Edge_index Type ERROR!')


def maybe_num_nodes_dict(edge_index_dict, num_nodes_dict=None):
    num_nodes_dict = {} if num_nodes_dict is None else copy(num_nodes_dict)

    found_types = list(num_nodes_dict.keys())

    for keys, edge_index in edge_index_dict.items():

        key = keys[0]
        if key not in found_types:
            N = int(tlx.reduce_max(edge_index[0], axis=0) + 1)
            num_nodes_dict[key] = max(N, num_nodes_dict.get(key, N))

        key = keys[-1]
        if key not in found_types:
            N = int(tlx.reduce_max(edge_index[1], axis=0) + 1)
            num_nodes_dict[key] = max(N, num_nodes_dict.get(key, N))

    return num_nodes_dict
