# -*- coding: utf-8 -*-
# @author WuJing
# @created 2023/4/10
from dataclasses import dataclass
from time import time

import numpy as np
import tensorlayerx as tlx
from typing import Union, List, Optional, Callable, Tuple

from gammagl.sparse import SparseGraph
from gammagl.utils.platform_utils import Tensor, all_to_tensor, to_list


@dataclass
class EdgeIndex:
    edge_index: Tensor
    e_id: Optional[Tensor]
    size: Tuple[int, int]


@dataclass
class Adj:
    adj_t: SparseGraph
    e_id: Optional[Tensor]
    size: Tuple[int, int]


class NeighborSampler(tlx.dataflow.DataLoader):
    def __init__(self, edge_index: Union[Tensor, SparseGraph],
                 sample_lists: List[int], node_idx: Optional[Tensor] = None,
                 num_nodes: Optional[int] = None, return_e_id: bool = True,
                 transform: Callable = None, **kwargs):
        self.edge_index = edge_index
        self.node_idx = node_idx
        self.num_nodes = num_nodes

        self.sizes = sample_lists
        self.return_e_id = return_e_id
        self.transform = transform
        self.is_sparse_graph = isinstance(edge_index, SparseGraph)
        self.__val__ = None

        if not self.is_sparse_graph:
            if (num_nodes is None and node_idx is not None
                    and node_idx.dtype == tlx.bool):
                num_nodes = node_idx.shape[0]
            if (num_nodes is None and node_idx is not None
                    and node_idx.dtype == tlx.int64):
                num_nodes = max(int(tlx.reduce_max(edge_index)), int(tlx.reduce_max(node_idx))) + 1
            if num_nodes is None:
                num_nodes = int(tlx.reduce_max(edge_index)) + 1

            value = tlx.arange(edge_index.shape[1]) if return_e_id else None
            self.adj_t = SparseGraph(row=edge_index[0], col=edge_index[1],
                                     value=value,
                                     sparse_sizes=(num_nodes, num_nodes)).t()

        else:
            adj_t = edge_index
            if return_e_id:
                self.__val__ = adj_t.storage.value()
                value = tlx.arange(adj_t.nnz())
                adj_t = adj_t.set_value(value, layout='coo')
            self.adj_t = adj_t

        self.adj_t.storage.rowptr()

        if node_idx is None:
            node_idx = tlx.arange(self.adj_t.sparse_size(0))
        elif node_idx.dtype == tlx.bool:
            node_idx = tlx.convert_to_tensor(np.reshape(tlx.convert_to_numpy(node_idx).nonzero(), -1))

        super().__init__(to_list(tlx.reshape(node_idx, -1)), collate_fn=self.sample, **kwargs)

    def sample(self, batch):
        if not isinstance(batch, Tensor):
            batch = all_to_tensor(batch)

        batch_size: int = len(batch)

        adjs = []
        n_id = batch


        for size in self.sizes:
            adj_t, n_id = self.adj_t.sample_adj(n_id, size, replace=False)

            e_id = adj_t.storage.value()
            size = adj_t.sparse_sizes()[::-1]
            if self.__val__ is not None:
                # adj_t.set_value_(self.__val__[e_id], layout='coo')
                adj_t.set_value_(tlx.gather(self.__val__, e_id), layout='coo')

            if self.is_sparse_graph:
                adjs.append(Adj(adj_t, e_id, size))

            else:
                row, col, _ = adj_t.coo()

                edge_index = tlx.stack([col, row], axis=0)
                adjs.append(EdgeIndex(edge_index, e_id, size))


        adjs = adjs[0] if len(adjs) == 1 else adjs[::-1]
        out = (batch, n_id, adjs)
        out = self.transform(*out) if self.transform is not None else out

        return out

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(sizes={self.sizes})'
