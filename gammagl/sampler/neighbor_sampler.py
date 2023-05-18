# -*- coding: utf-8 -*-
# @author WuJing
# @created 2023/3/2
from typing import List, Dict, Optional, Any
import numpy as np
import gammagl
from gammagl.loader.utils import to_hetero_csc
from gammagl.loader.utils import to_csc, remap_keys
from gammagl.sampler.base_sampler import BaseSampler
from gammagl.data.graph import Graph
from gammagl.data.heterograph import HeteroGraph
import tensorlayerx as tlx
from dataclasses import dataclass
import random
import gammagl.ops
from gammagl.typing import NodeType
from gammagl.utils.platform_utils import Tensor, EdgeType


def add_negative_samples(
        edge_label_index,
        edge_label,
        edge_label_time,
        num_src_nodes: int,
        num_dst_nodes: int,
        negative_sampling_ratio: float,
):
    """Add negative samples and their `edge_label` and `edge_time`
    if `neg_sampling_ratio > 0`"""
    # num_pos_edges = edge_label_index.size(1)
    num_pos_edges = edge_label_index.shape[1]
    num_neg_edges = int(num_pos_edges * negative_sampling_ratio)

    if num_neg_edges == 0:
        return edge_label_index, edge_label, edge_label_time

    neg_row = tlx.convert_to_tensor(np.full((num_neg_edges,), random.randint(0, num_src_nodes)))
    neg_col = tlx.convert_to_tensor(np.full((num_neg_edges,), random.randint(0, num_dst_nodes)))

    neg_edge_label_index = tlx.stack([neg_row, neg_col], axis=0)

    edge_label_index = tlx.concat([
        edge_label_index,
        neg_edge_label_index,
    ], axis=1)

    pos_edge_label = edge_label + 1

    neg_edge_label = tlx.zeros((num_neg_edges,) + edge_label.shape[1:], dtype=edge_label.dtype,
                               device=edge_label.device)

    tlx.zeros(edge_label.shape, dtype=edge_label.dtype, device=edge_label.device)

    edge_label = tlx.concat([pos_edge_label, neg_edge_label], axis=0)

    return edge_label_index, edge_label, edge_label_time


class NeighborSampler(BaseSampler):
    def __init__(self, graph, num_neighbors: List[int], replace=False, directed=True, input_type=None, is_sorted=False):
        self.data_cls = graph.__class__ if isinstance(
            graph, (Graph, HeteroGraph)) else 'custom'
        self.graph = graph
        self.num_neighbors = num_neighbors
        self.replace = replace
        self.directed = True

        self.num_src_nodes, self.num_dst_nodes = None, None
        if isinstance(graph, Graph):
            self.num_src_nodes = self.num_dst_nodes = graph.num_nodes

        if isinstance(graph, Graph):
            out = to_csc(graph, device='cpu', is_sorted=is_sorted)
            self.colptr, self.row, self.perm = out
            assert isinstance(num_neighbors, (list, tuple))
        elif isinstance(graph, HeteroGraph):
            self.node_types, self.edge_types = graph.metadata()
            self._set_num_neighbors_and_num_hops(num_neighbors)
            assert input_type is not None
            self.input_type = input_type
            out = to_hetero_csc(graph, device='cpu',
                                is_sorted=is_sorted)
            colptr_dict, row_dict, perm_dict = out

            self.to_rel_type = {key: '__'.join(key) for key in self.edge_types}
            self.to_edge_type = {
                '__'.join(key): key
                for key in self.edge_types
            }
            self.row_dict = remap_keys(row_dict, self.to_rel_type)
            self.colptr_dict = remap_keys(colptr_dict, self.to_rel_type)
            self.num_neighbors = remap_keys(self.num_neighbors, self.to_rel_type)
            self.perm = perm_dict
        else:
            raise TypeError(f"'{self.__class__.__name__}'' found invalid "
                            f"type: '{type(graph)}'")

    def _set_num_neighbors_and_num_hops(self, num_neighbors):
        if isinstance(num_neighbors, (list, tuple)):
            num_neighbors = {key: num_neighbors for key in self.edge_types}
        assert isinstance(num_neighbors, dict)
        self.num_neighbors = num_neighbors
        self.num_hops = max([0] + [len(v) for v in num_neighbors.values()])

        for key, value in self.num_neighbors.items():
            if len(value) != self.num_hops:
                raise ValueError(f"Expected the edge type {key} to have "
                                 f"{self.num_hops} entries (got {len(value)})")

    def _sample(self, seed, **kwargs):
        if issubclass(self.data_cls, Graph):
            out = gammagl.ops.sparse.neighbor_sample(
                self.colptr,
                self.row,
                seed,  # seed
                self.num_neighbors,
                self.replace,
                self.directed,
            )

            node, row, col, edge = out
            batch = None

            return SamplerOutput(
                node=node,
                row=row,
                col=col,
                edge=edge,
                batch=batch)

        elif issubclass(self.data_cls, HeteroGraph):

            out = gammagl.ops.sparse.hetero_neighbor_sample(
                self.node_types,
                self.edge_types,
                self.colptr_dict,
                self.row_dict,
                seed,  # seed_dict
                self.num_neighbors,
                self.num_hops,
                self.replace,
                self.directed,
            )
            node, row, col, edge = out
            batch = None

            return HeteroSamplerOutput(node=node,
                                       row=remap_keys(row, self.to_edge_type),
                                       col=remap_keys(col, self.to_edge_type),
                                       edge=remap_keys(edge, self.to_edge_type),
                                       batch=batch)

        else:
            raise TypeError(f"'{self.__class__.__name__}'' found invalid "
                            f"type: '{self.data_cls}'")

    def sample_from_nodes(self, index, **kwargs):

        if isinstance(index, (list, tuple)):
            index = tlx.convert_to_tensor(index)

        if issubclass(self.data_cls, Graph):
            output = self._sample(seed=index)
            output.metadata = tlx.numel(index)

        elif issubclass(self.data_cls, HeteroGraph):
            output = self._sample(seed={self.input_type: index})
            output.metadata = tlx.numel(index)

        else:
            raise TypeError(f"'{self.__class__.__name__}'' found invalid "
                            f"type: '{self.data_cls}'")

        return output

    def sample_from_edges(self, index, **kwargs):
        negative_sampling_ratio = kwargs.get('negative_sampling_ratio', 0.0)
        query = [tlx.stack(s, axis=0) for s in zip(*index)]
        edge_label_index = tlx.stack(query[:2], axis=0)
        edge_label = query[2]
        edge_label_time = query[3] if len(query) == 4 else None

        out = add_negative_samples(edge_label_index, edge_label,
                                   edge_label_time, self.num_src_nodes,
                                   self.num_dst_nodes, negative_sampling_ratio)
        edge_label_index, edge_label, edge_label_time = out
        # orig_edge_label_index = edge_label_index
        if issubclass(self.data_cls, Graph):

            query_nodes = tlx.reshape(edge_label_index, -1)

            query_nodes, reverse = gammagl.ops.unique(query_nodes, return_inverse=True)

            edge_label_index = tlx.reshape(reverse, (2, -1))

            output = self._sample(seed=query_nodes)
            output.metadata = (edge_label_index, edge_label)
        else:
            raise TypeError(f"'{self.__class__.__name__}'' found invalid "
                            f"type: '{self.data_cls}'")

        return output


@dataclass
class SamplerOutput:
    node: Tensor
    row: Tensor
    col: Tensor
    edge: Tensor
    batch: Optional[Tensor] = None
    metadata: Optional[Any] = None


@dataclass
class HeteroSamplerOutput:
    node: Dict[NodeType, Tensor]
    row: Dict[EdgeType, Tensor]
    col: Dict[EdgeType, Tensor]
    edge: Dict[EdgeType, Tensor]
    batch: Optional[Dict[NodeType, Tensor]] = None
    metadata: Optional[Any] = None
