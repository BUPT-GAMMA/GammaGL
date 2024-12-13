# -*- coding: utf-8 -*-
# @author WuJing
# @created 2023/4/10
import tensorlayerx as tlx
from gammagl.sparse import SparseGraph
from gammagl.sparse.storage import SparseStorage


def t(src: SparseGraph) -> SparseGraph:
    csr2csc = src.storage.csr2csc()

    row, col, value = src.coo()

    if value is not None:
        value = tlx.gather(value, csr2csc)

    sparse_sizes = src.storage.sparse_sizes()

    storage = SparseStorage(
        row=tlx.gather(col, csr2csc),
        rowptr=src.storage._colptr,
        col=tlx.gather(row, csr2csc),
        value=value,
        sparse_sizes=(sparse_sizes[1], sparse_sizes[0]),
        rowcount=src.storage._colcount,
        colptr=src.storage._rowptr,
        colcount=src.storage._rowcount,
        csr2csc=src.storage._csc2csr,
        csc2csr=csr2csc,
        is_sorted=True,
    )

    return src.from_storage(storage)


SparseGraph.t = lambda self: t(self)
