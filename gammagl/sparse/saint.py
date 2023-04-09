# -*- coding: utf-8 -*-
# @author WuJing
# @created 2023/4/4

from ._saint import c_saint_subgraph
from ..utils.platform_utils import ops_func


@ops_func
def saint_subgraph(node_idx, rowptr, row, col):
    return c_saint_subgraph(node_idx, rowptr, row, col)
