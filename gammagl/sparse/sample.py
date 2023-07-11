# -*- coding: utf-8 -*-
# @author WuJing
# @created 2023/4/10
from time import time
from typing import Tuple

import tensorlayerx as tlx
import gammagl.ops
from gammagl.sparse import SparseGraph
from gammagl.utils.platform_utils import Tensor


def sample_adj(src: SparseGraph, subset: Tensor, num_neighbors: int,
               replace: bool = False) -> Tuple[SparseGraph, Tensor]:
    rowptr, col, value = src.csr()

    # start = time()
    rowptr, col, n_id, e_id = gammagl.ops.sparse.sample_adj(
        rowptr, col, subset, num_neighbors, replace)
    # print(f'c算子 cost {time()-start}s')
    if value is not None:
        value = tlx.gather(value, e_id)

    out = SparseGraph(rowptr=rowptr, row=None, col=col, value=value,
                      sparse_sizes=(subset.shape[0], n_id.shape[0]),
                      is_sorted=True)

    return out, n_id


SparseGraph.sample_adj = sample_adj
