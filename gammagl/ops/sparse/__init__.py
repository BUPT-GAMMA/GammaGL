# -*- coding: utf-8 -*-
# @author WuJing
# @created 2023/4/11

try:
    from .sparse import (
    ind2ptr,
    ptr2ind,
    neighbor_sample,
    hetero_neighbor_sample,
    saint_subgraph,
    random_walk,
    sample_adj
)
except:
    Warning("sparse ops load failed.")