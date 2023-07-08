# -*- coding: utf-8 -*-
# @author WuJing
# @created 2023/4/3


import tensorlayerx as tlx
from tensorlayerx.dataflow import DataLoader
from gammagl.data import Graph
from gammagl.loader.utils import get_input_nodes_index
from gammagl.sparse.graph import SparseGraph
from gammagl.ops import unique
import numpy as np


class GraphSAINTSampler(DataLoader):
    def __init__(self, graph: Graph, batch_size: int, num_steps: int = 1,
                 sample_coverage: int = 0, **kwargs):
        self.num_steps = num_steps
        self.__batch_size__ = batch_size

        self.N = N = graph.num_nodes
        self.E = graph.num_edges

        self.adj = SparseGraph(
            row=graph.edge_index[0], col=graph.edge_index[1],
            value=tlx.arange(self.E),
            sparse_sizes=(N, N)
        )

        self.graph = graph

        _, input_nodes_index = get_input_nodes_index(graph, None)
        super().__init__(self, batch_size=1, collate_fn=self.__collate__, **kwargs)

    def __len__(self):
        return self.num_steps

    def __sample_nodes__(self, batch_size):
        raise NotImplementedError

    def __getitem__(self, idx):
        node_idx = unique(self.__sample_nodes__(self.__batch_size__))
        adj, _ = self.adj.saint_subgraph(node_idx)
        return node_idx, adj

    def __collate__(self, data_list):
        assert len(data_list) == 1
        node_idx, adj = data_list[0]

        row, col, edge_idx = adj.coo()
        graph = Graph(x=None, num_nodes=node_idx.shape[0], edge_index=tlx.stack([row, col], axis=0))

        for key, item in self.graph.iter():
            if key in ['edge_index', 'num_nodes']:
                continue
            if tlx.is_tensor(item) and item.shape[0] == self.N:
                graph[key] = tlx.gather(item, node_idx)
            elif tlx.is_tensor(item) and item.shape[0] == self.E:
                graph[key] = tlx.gather(item, edge_idx)
            else:
                graph[key] = item

        return graph


class GraphSAINTNodeSampler(GraphSAINTSampler):
    r"""The GraphSAINT node sampler class (see
    :class:`~torch_geometric.loader.GraphSAINTSampler`).
    """

    def __sample_nodes__(self, batch_size):
        edge_sample = tlx.convert_to_tensor(np.random.randint(0, self.E, (batch_size, self.batch_size), dtype=int))
        return self.adj.storage.row()[edge_sample]


class GraphSAINTRandomWalkSampler(GraphSAINTSampler):
    def __init__(self, data, batch_size: int, walk_length: int,
                 num_steps: int = 1, sample_coverage: int = 0, **kwargs):
        self.walk_length = walk_length
        super().__init__(data, batch_size, num_steps, sample_coverage, **kwargs)

    def __sample_nodes__(self, batch_size):
        start = np.random.randint(0, self.N, (batch_size,))
        node_idx = self.adj.random_walk(tlx.convert_to_tensor(start.flatten()), self.walk_length)
        return tlx.reshape(node_idx, -1)
