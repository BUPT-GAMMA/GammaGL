# -*- coding: utf-8 -*-
# @author WuJing
# @created 2023/4/10

import tensorlayerx as tlx

from gammagl.sparse.storage import SparseStorage


def coalesce(index, value, m, n, op="add"):
    storage = SparseStorage(row=index[0], col=index[1], value=value,
                            sparse_sizes=(m, n), is_sorted=False)
    storage = storage.coalesce(reduce=op)
    return tlx.stack([storage.row(), storage.col()], axis=0), storage.value()
