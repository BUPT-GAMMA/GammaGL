# -*- coding: utf-8 -*-
# @author WuJing
# @created 2023/3/23

from typing import List, Tuple, Union

from ._convert import c_set_to_one, c_ind2ptr, c_ptr2ind
from ._neighbor_sample import c_neighbor_sample,c_hetero_neighbor_sample

# neighbor_sample as c_neighbor_sample, \
# ind2ptr as c_ind2ptr
from gammagl.utils.platform_utils import ops_func


def set_to_one(arr, num_worker=0):
    return c_set_to_one(arr, num_worker)


@ops_func
def neighbor_sample(colptr, row, seed, num_neighbors, replace=False, dircted=True):
    return c_neighbor_sample(colptr, row, seed, num_neighbors, replace, dircted)


@ops_func
def hetero_neighbor_sample(node_types: Union[List, Tuple],
                           edge_types: Union[List[Union[List, Tuple]], Tuple[Union[List, Tuple]]],
                           col_dict,
                           row_dict,
                           input_node_dict,
                           num_neighbors_dict,
                           num_hops,
                           replace,
                           directed
                           ):
    return c_hetero_neighbor_sample(node_types, edge_types, col_dict, row_dict, input_node_dict, num_neighbors_dict,
                                    num_hops, replace, directed)


@ops_func
def ind2ptr(ind, M, num_worker=0):
    return c_ind2ptr(ind, M, num_worker)


@ops_func
def ptr2ind(ptr, E, num_worker=0):
    return c_ptr2ind(ptr, E, num_worker)

# @ops_func
# def random_walk(rowptr, col, start, walk_length):
#     return c_random_walk(rowptr, col, start, walk_length)
