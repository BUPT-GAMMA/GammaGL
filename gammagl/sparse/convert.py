# -*- coding: utf-8 -*-
# @author WuJing
# @created 2023/3/23
import numpy as np
from typing import List, Tuple, Union

from ._convert import c_set_to_one, c_ind2ptr, c_neighbor_sample, c_hetero_neighbor_sample


# neighbor_sample as c_neighbor_sample, \
# ind2ptr as c_ind2ptr


def set_to_one(arr, num_worker=0):
    return c_set_to_one(arr, num_worker)


def neighbor_sample(colptr, row, seed, num_neighbors, replace=False, dircted=True):
    return c_neighbor_sample(colptr, row, seed, num_neighbors, replace, dircted)


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


def ind2ptr(ind, M, num_worker=0):
    return c_ind2ptr(ind, M, num_worker)
