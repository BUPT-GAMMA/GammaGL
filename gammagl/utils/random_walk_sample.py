# -*- coding: utf-8 -*-
# @author WuJing
# @created 2023/4/1


import random
import numba
import numpy as np

# from gammagl.loader.utils import to_csr
from gammagl.data import Graph
from gammagl.loader.utils import to_csr
from gammagl.utils.platform_utils import all_to_numpy


@numba.njit(cache=True)
def _random_walk(node, length, indptr, indices, p=0.0):
    result = [numba.int32(0)] * length
    result[0] = numba.int32(node)
    i = numba.int32(1)
    _node = node
    _start = indptr[_node]
    _end = indptr[_node + 1]
    while i < length:
        start = indptr[node]
        end = indptr[node + 1]
        sample = random.randint(start, end - 1)
        node = indices[sample]
        if np.random.uniform(0, 1) > p:
            result[i] = node
        else:
            sample = random.randint(_start, _end - 1)
            node = indices[sample]
            result[i] = node
        i += 1
    return np.array(result, dtype=np.int32)


@numba.njit(cache=True, parallel=True)
def random_walk_parallel(start, length, indptr, indices, p):
    result = [np.zeros(length, dtype=np.int32)] * len(start)
    for i in numba.prange(len(start)):
        result[i] = _random_walk(start[i], length, indptr, indices, p)
    return result


def rw_sample(graph, start, walk_length, p=0):
    indptr, indices, perm = to_csr(graph, None, False)
    indptr = all_to_numpy(indptr)
    indices = all_to_numpy(indices)
    result = random_walk_parallel(start, walk_length, indptr, indices, p)
    return result


def rw_sample_by_edge_index(edge_index, start, walk_length, p=0):
    graph = Graph(edge_index=edge_index)
    return rw_sample(graph, start, walk_length, p)
