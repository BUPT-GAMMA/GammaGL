# -*- coding: utf-8 -*-
# @author WuJing
# @created 2023/4/4

import tensorlayerx as tlx
from typing import Optional, Tuple, List

from textwrap import indent

from gammagl.sparse.storage import SparseStorage
from gammagl.utils.platform_utils import Tensor


class SparseGraph(object):
    storage: SparseStorage

    def __init__(
            self,
            row=None,
            rowptr=None,
            col=None,
            value=None,
            sparse_sizes: Optional[Tuple[Optional[int], Optional[int]]] = None,
            is_sorted: bool = False,
            trust_data: bool = False,
    ):
        self.storage = SparseStorage(
            row=row,
            rowptr=rowptr,
            col=col,
            value=value,
            sparse_sizes=sparse_sizes,
            rowcount=None,
            colptr=None,
            colcount=None,
            csr2csc=None,
            csc2csr=None,
            is_sorted=is_sorted,
            trust_data=trust_data,
        )

    def coo(self) -> Tuple:
        return self.storage.row(), self.storage.col(), self.storage.value()

    def csr(self) -> Tuple:
        return self.storage.rowptr(), self.storage.col(), self.storage.value()

    def csc(self) -> Tuple:
        perm = self.storage.csr2csc()
        value = self.storage.value()
        if value is not None:
            value = tlx.gather(value, perm)
        return self.storage.colptr(), tlx.gather(self.storage.row(), perm), value

    @classmethod
    def from_edge_index(
            self,
            edge_index,
            edge_attr=None,
            sparse_sizes: Optional[Tuple[Optional[int], Optional[int]]] = None,
            is_sorted: bool = False,
            trust_data: bool = False,
    ):
        return SparseGraph(row=edge_index[0], rowptr=None, col=edge_index[1],
                           value=edge_attr, sparse_sizes=sparse_sizes,
                           is_sorted=is_sorted, trust_data=trust_data)

    @classmethod
    def from_storage(self, storage: SparseStorage):
        out = SparseGraph(
            row=storage._row,
            rowptr=storage._rowptr,
            col=storage._col,
            value=storage._value,
            sparse_sizes=storage._sparse_sizes,
            is_sorted=True,
            trust_data=True,
        )
        out.storage._rowcount = storage._rowcount
        out.storage._colptr = storage._colptr
        out.storage._colcount = storage._colcount
        out.storage._csr2csc = storage._csr2csc
        out.storage._csc2csr = storage._csc2csr
        return out

    def sparse_sizes(self) -> Tuple[int, int]:
        return self.storage.sparse_sizes()

    def sparse_size(self, dim: int) -> int:
        return self.storage.sparse_sizes()[dim]

    def sizes(self) -> List[int]:
        sparse_sizes = self.sparse_sizes()
        value = self.storage.value()
        if value is not None:
            # return list(sparse_sizes) + list(value.size())[1:]
            return list(sparse_sizes) + list(value.shape)[1:]
        else:
            return list(sparse_sizes)

    def nnz(self) -> int:
        return tlx.numel(self.storage.col())

    def set_value_(self, value: Optional[Tensor],
                   layout: Optional[str] = None):
        self.storage.set_value_(value, layout)
        return self

    def set_value(self, value: Optional[Tensor],
                  layout: Optional[str] = None):
        return self.from_storage(self.storage.set_value(value, layout))

    def __repr__(self) -> str:
        i = ' ' * 6
        row, col, value = self.coo()
        infos = []
        infos += [f'row={indent(row.__repr__(), i)[len(i):]}']
        infos += [f'col={indent(col.__repr__(), i)[len(i):]}']

        if value is not None:
            infos += [f'val={indent(value.__repr__(), i)[len(i):]}']

        infos += [
            f'size={tuple(self.sizes())}, nnz={self.nnz()}'
            # f'size={tuple(self.sizes())}, nnz={self.nnz()}, '
            # f'density={100 * self.density():.02f}%'
        ]

        infos = ',\n'.join(infos)

        i = ' ' * (len(self.__class__.__name__) + 1)
        return f'{self.__class__.__name__}({indent(infos, i)[len(i):]})'
