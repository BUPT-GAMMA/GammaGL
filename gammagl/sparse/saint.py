# -*- coding: utf-8 -*-
# @author WuJing
# @created 2023/4/4
from typing import Tuple
import tensorlayerx as tlx
import gammagl
from gammagl.sparse import SparseGraph


def saint_subgraph(src: SparseGraph, node_idx) -> Tuple:
    row, col, value = src.coo()
    rowptr = src.storage.rowptr()

    # data = torch.ops.torch_sparse.saint_subgraph(node_idx, rowptr, row, col)
    data = gammagl.ops.sparse.saint_subgraph(node_idx, rowptr, row, col)
    row, col, edge_index = data

    if value is not None:
        value = tlx.gather(value, edge_index)

    out = SparseGraph(row=row, rowptr=None, col=col, value=value,
                      sparse_sizes=(node_idx.shape[0], node_idx.shape[0]),
                      is_sorted=True)
    return out, edge_index


SparseGraph.saint_subgraph = saint_subgraph
