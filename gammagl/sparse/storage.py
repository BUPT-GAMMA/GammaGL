# -*- coding: utf-8 -*-
# @author WuJing
# @created 2023/4/4
import warnings

import numpy as np
import tensorlayerx as tlx
from typing import Tuple, Optional, List, Union
import gammagl
from gammagl.utils.platform_utils import all_to_numpy, Tensor


def get_layout(layout: Optional[str] = None) -> str:
    if layout is None:
        layout = 'coo'
        warnings.warn('`layout` argument unset, using default layout '
                      '"coo". This may lead to unexpected behaviour.')
    assert layout == 'coo' or layout == 'csr' or layout == 'csc'
    return layout


class SparseStorage:
    _row = None
    _rowptr = None
    _rowcount = None
    _col = None
    _colptr = None
    _colcount = None
    _value = None
    _sparse_sizes = None
    _csr2csc = None
    _csc2csr = None

    def __init__(
            self,
            row: Tensor = None,
            rowptr: Tensor = None,
            col: Tensor = None,
            value: Tensor = None,
            sparse_sizes: Union[Tensor, List, Tuple] = None,
            rowcount: Tensor = None,
            colptr: Tensor = None,
            colcount: Tensor = None,
            csr2csc: Tensor = None,
            csc2csr: Tensor = None,
            is_sorted: bool = False,
            trust_data: bool = False,
    ):

        assert row is not None or rowptr is not None
        assert col is not None
        # assert col.dtype == tlx.int64

        M: int = 0
        if sparse_sizes is None or sparse_sizes[0] is None:
            if rowptr is not None:
                M = tlx.numel(rowptr) - 1
            elif row is not None and tlx.numel(row) > 0:
                M = int(tlx.reduce_max(row)) + 1
        else:
            _M = sparse_sizes[0]
            assert _M is not None
            M = _M
            if rowptr is not None:
                assert tlx.numel(rowptr) - 1 == M
            elif row is not None and tlx.numel(row) > 0:
                assert trust_data or int(tlx.reduce_max(row)) < M

        N: int = 0
        if sparse_sizes is None or sparse_sizes[1] is None:
            if tlx.numel(col) > 0:
                N = int(tlx.reduce_max(col)) + 1
        else:
            _N = sparse_sizes[1]
            assert _N is not None
            N = _N
            if tlx.numel(col) > 0:
                assert trust_data or int(tlx.reduce_max(col)) < N

        sparse_sizes = (M, N)

        if row is not None:
            assert row.dtype == tlx.int64
            assert tlx.numel(row) == tlx.numel(col)

        if rowptr is not None:
            assert rowptr.dtype == tlx.int64
            assert tlx.numel(rowptr) - 1 == sparse_sizes[0]

        if value is not None:
            assert value.shape[0] == col.shape[0]

        if rowcount is not None:
            assert rowcount.dtype == tlx.int64
            assert tlx.numel(rowcount) == sparse_sizes[0]

        if colptr is not None:
            assert colptr.dtype == tlx.int64
            assert tlx.numel(colptr) - 1 == sparse_sizes[1]

        if colcount is not None:
            assert colcount.dtype == tlx.int64
            assert tlx.numel(colcount) == sparse_sizes[1]

        if csr2csc is not None:
            # assert csr2csc.dtype == tlx.int64
            assert tlx.numel(csr2csc) == col.shape[0]

        if csc2csr is not None:
            # assert csc2csr.dtype == tlx.int64
            assert tlx.numel(csc2csr) == col.shape[0]

        self._row = row
        self._rowptr = rowptr
        self._col = col
        self._value = value
        self._sparse_sizes = tuple(sparse_sizes)
        self._rowcount = rowcount
        self._colptr = colptr
        self._colcount = colcount
        self._csr2csc = csr2csc
        self._csc2csr = csc2csr

        if not is_sorted:
            
            idx = all_to_numpy(
                tlx.zeros(shape = (tlx.numel(self._col) + 1, ),
                          dtype=self._col.dtype)
                )
            idx[1:] = all_to_numpy(self.row())
            idx[1:] *= all_to_numpy(self._sparse_sizes[1])
            idx[1:] += all_to_numpy(self._col)
            if np.any(idx[1:] < idx[:-1]):
                perm = np.argsort(idx[1:])
                self._row = tlx.gather(self.row(), perm)
                self._col = tlx.gather(self._col, perm)
                if value is not None:
                    self._value = tlx.gather(value, perm)
                self._csr2csc = None
                self._csc2csr = None

    def row(self):
        row = self._row
        if row is not None:
            return row

        rowptr = self._rowptr
        if rowptr is not None:
            row = gammagl.ops.sparse.ptr2ind(rowptr, tlx.numel(self._col))
            self._row = row
            return row

        raise ValueError

    def col(self):
        return self._col

    def rowptr(self):
        rowptr = self._rowptr
        if rowptr is not None:
            return rowptr

        row = self._row
        if row is not None:
            rowptr = gammagl.ops.sparse.ind2ptr(row, self._sparse_sizes[0])
            self._rowptr = rowptr
            return rowptr

        raise ValueError

    def colptr(self):
        colptr = self._colptr
        if colptr is not None:
            return colptr

        csr2csc = self._csr2csc
        if csr2csc is not None:
            colptr = gammagl.ops.sparse.ind2ptr(self._col[csr2csc], self._sparse_sizes[1])
        else:
            colptr = tlx.zeros(self._sparse_sizes[1] + 1, dtype=self._col.dtype)
            tlx.cumsum(self.colcount(), axis=0, out=colptr[1:])
        self._colptr = colptr
        return colptr

    def colcount(self):
        colcount = self._colcount
        if colcount is not None:
            return colcount

        colptr = self._colptr
        if colptr is not None:
            colcount = colptr[1:] - colptr[:-1]
        else:
            # TODO maybe wrong
            # colcount = scatter_add(tlx.ones_like(self._col), self._col, dim_size=self._sparse_sizes[1])
            colcount = tlx.segment_sum(self._col, tlx.ones_like(self._col))
        self._colcount = colcount
        return colcount

    def csr2csc(self):
        csr2csc = self._csr2csc
        if csr2csc is not None:
            return csr2csc

        idx = self._sparse_sizes[0] * self._col + self.row()
        csr2csc = tlx.argsort(idx)
        self._csr2csc = csr2csc
        return csr2csc

    def csc2csr(self):
        csc2csr = self._csc2csr
        if csc2csr is not None:
            return csc2csr

        csc2csr = tlx.argsort(self.csr2csc())
        self._csc2csr = csc2csr
        return csc2csr

    def set_value_(self, value: Optional[Tensor],
                   layout: Optional[str] = None):
        if value is not None:
            if get_layout(layout) == 'csc':
                value = tlx.gather(value, self.csc2csr())
            assert value.shape[0] == tlx.numel(self._col)

        self._value = value
        return self

    def set_value(self, value: Optional[Tensor],
                  layout: Optional[str] = None):
        if value is not None:
            if get_layout(layout) == 'csc':
                value = tlx.gather(value, self.csc2csr())
            assert value.shape[0] == tlx.numel(self._col)

        return SparseStorage(
            row=self._row,
            rowptr=self._rowptr,
            col=self._col,
            value=value,
            sparse_sizes=self._sparse_sizes,
            rowcount=self._rowcount,
            colptr=self._colptr,
            colcount=self._colcount,
            csr2csc=self._csr2csc,
            csc2csr=self._csc2csr,
            is_sorted=True,
            trust_data=True,
        )

    def value(self):
        return self._value

    def sparse_sizes(self) -> Tuple[int, int]:
        return self._sparse_sizes

    def sparse_size(self, dim: int) -> int:
        return self._sparse_sizes[dim]

    # ops
    def coalesce(self, reduce: str = "add"):
        idx = np.full((tlx.numel(self._col) + 1), -1, dtype=np.int64)

        idx[1:] = self._sparse_sizes[1] * self.row() + self._col
        mask = idx[1:] > idx[:-1]

        if np.all(mask):  # Skip if indices are already coalesced.
            return self

        try:
            row = tlx.gather(self.row(), mask)
            col = tlx.gather(self._col, mask)
        except:
            row = tlx.gather(self.row(), np.where(mask))
            col = tlx.gather(self._col, np.where(mask))

        # TODO Not implemented
        value = self._value
        # if value is not None:
        #     ptr = mask.nonzero().flatten()
        #     ptr = tlx.concat([ptr, ptr.new_full((1,), value.size(0))])
        #     value = segment_csr(value, ptr, reduce=reduce)

        return SparseStorage(
            row=row,
            rowptr=None,
            col=col,
            value=value,
            sparse_sizes=self._sparse_sizes,
            rowcount=None,
            colptr=None,
            colcount=None,
            csr2csc=None,
            csc2csr=None,
            is_sorted=True,
            trust_data=True,
        )
