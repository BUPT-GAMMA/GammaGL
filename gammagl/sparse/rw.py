# -*- coding: utf-8 -*-
# @author WuJing
# @created 2023/4/5

import gammagl
from gammagl.sparse import SparseGraph


def random_walk(src: SparseGraph, start,
                walk_length: int):
    rowptr, col, _ = src.csr()
    # return torch.ops.torch_sparse.random_walk(rowptr, col, start, walk_length)
    return gammagl.ops.sparse.random_walk(rowptr, col, start, walk_length)


SparseGraph.random_walk = random_walk
