# -*- coding: utf-8 -*-
# @author WuJing
# @created 2023/4/5
# SparseGraph local method ops registry
from time import time
from typing import List, Dict, Tuple, Union
import tensorlayerx as tlx

# try for development
try:
    from ._sparse import (
        c_sample_adj,
        c_random_walk,
        c_saint_subgraph,
        c_ind2ptr,
        c_ptr2ind,
        c_neighbor_sample,
        c_hetero_neighbor_sample
    )
except:
    from ._sample import c_sample_adj
    from ._rw import c_random_walk
    from ._saint import c_saint_subgraph
    from ._convert import c_ind2ptr, c_ptr2ind
    from ._neighbor_sample import c_neighbor_sample, c_hetero_neighbor_sample

from ._sparse_cuda import cuda_torch_ind2ptr, cuda_torch_ptr2ind, cuda_torch_neighbor_sample, cuda_torch_sample_adj

from gammagl.utils.platform_utils import out_tensor_list, Tensor, out_tensor



@out_tensor
def ind2ptr(
        ind: Tensor,
        M: int,
        num_worker: int = 0) -> Tensor:
    if(str(ind.device)=='cpu'):
        return c_ind2ptr(ind, M, num_worker)
    else:
        return cuda_torch_ind2ptr(ind, M)

# @out_tensor
# def cu_ind2ptr(
#         ind: Tensor,
#         M: int,
#         num_worker: int = 0) -> Tensor:
#     return cuda_ind2ptr(ind, M, num_worker)


@out_tensor
def ptr2ind(
        ptr: Tensor,
        E: int,
        num_worker: int = 1):
    if(str(ptr.device)=='cpu'):
        return c_ptr2ind(ptr, E, num_worker)
    else:
        return cuda_torch_ptr2ind(ptr, E)


@out_tensor_list
def neighbor_sample(
        colptr: Tensor,
        row: Tensor,
        input_node: Tensor,
        num_neighbors: List,
        replace: bool,
        directed: bool) -> Tuple[List, List, List, List]:
    if(str(colptr.device)=='cpu'):
        start = time()
        res = c_neighbor_sample(colptr, row, input_node, num_neighbors, replace, directed)
        print(f'c算子 cost {time() - start}s')
    else:
        start = time()
        num_neighbors = Tensor(num_neighbors).to('cpu')
        res = cuda_torch_neighbor_sample(colptr, row, input_node, num_neighbors, replace, directed, 0)
        print(f'cuda算子 cost {time() - start}s')
    return res


@out_tensor_list
def hetero_neighbor_sample(
        node_types: List[str],
        edge_types: List[Union[List[str], Tuple[str, str, str]]],
        colptr_dict: Dict[str, Tensor],
        row_dict: Dict[str, Tensor],
        input_node_dict: Dict[str, Tensor],
        num_neighbors_dict: Dict[str, Tensor],
        num_hops: int,
        replace=False,
        directed=False
) -> Tuple[Dict[str, List], Dict[str, List], Dict[str, List], Dict[str, List]]:
    return c_hetero_neighbor_sample(
        node_types,
        edge_types,
        colptr_dict,
        row_dict,
        input_node_dict,
        num_neighbors_dict,
        num_hops,
        replace,
        directed
    )


@out_tensor
def random_walk(
        rowptr: Tensor,
        col: Tensor,
        start: List,
        walk_length: int):
    return c_random_walk(rowptr, col, start, walk_length)


@out_tensor_list
def saint_subgraph(
        node_idx: Tensor,
        rowptr: Tensor,
        row: Tensor,
        col: Tensor):
    return c_saint_subgraph(node_idx, rowptr, row, col)


@out_tensor_list
def sample_adj(
        rowptr: Tensor,
        col: Tensor,
        idx: Tensor,
        num_neighbors: int,
        replace: bool = False):
    if(str(rowptr.device)=='cpu'):
        res = c_sample_adj(rowptr, col, idx, num_neighbors, replace)
    else:
        num_neighbors = tlx.convert_to_tensor([num_neighbors], dtype=tlx.int64).to('cpu')
        res = cuda_torch_sample_adj(rowptr, col, idx, num_neighbors, replace, False, 0)
    return res
