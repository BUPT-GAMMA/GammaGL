# -*- coding: utf-8 -*-
# @author WuJing
# @created 2023/4/5
import typing

import numpy as np
import tensorlayerx as tlx
from gammagl.data import Graph
from gammagl.loader.graph_saint import GraphSAINTRandomWalkSampler
from gammagl.utils.platform_utils import all_to_numpy, all_to_tensor


def non_zero(tensor):
    if tlx.is_tensor(tensor):
        arr = all_to_numpy(tensor).nonzero()
        return tlx.convert_to_tensor(arr)
    raise TypeError("Not a valid tensor!")


def test_graph_saint():
    adj = np.array([
        [+1, +2, +3, +0, +4, +0],
        [+5, +6, +0, +7, +0, +8],
        [+9, +0, 10, +0, 11, +0],
        [+0, 12, +0, 13, +0, 14],
        [15, +0, 16, +0, 17, +0],
        [+0, 18, +0, 19, +0, 20],
    ])
    edge_index = adj.nonzero()
    # TODO why
    edge_id = adj[edge_index[0], edge_index[1]]

    edge_index = all_to_tensor(edge_index)
    edge_id = all_to_tensor(edge_id)
    x = tlx.convert_to_tensor([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
    n_id = tlx.arange(6)
    graph = Graph(edge_index=edge_index, x=x, n_id=n_id, edge_id=edge_id,
                  num_nodes=6)

    loader = GraphSAINTRandomWalkSampler(graph, batch_size=2, walk_length=1,
                                         num_steps=4, shuffle=False)

    assert len(loader) == 4
    for sample in loader:
        assert isinstance(sample, Graph)
        n_e = sample.num_edges
        sample = all_to_numpy(sample)

        assert sample.num_nodes <= graph.num_nodes
        assert sample.n_id.min() >= 0 and sample.n_id.max() < 6
        assert sample.num_nodes == sample.n_id.shape[0]
        assert sample.x.tolist() == tlx.convert_to_numpy(x)[sample.n_id].tolist()
        assert sample.edge_index.min() >= 0
        assert sample.edge_index.max() < sample.num_nodes
        assert sample.edge_id.min() >= 1 and sample.edge_id.max() <= 21

        # TODO 是个问题
        # assert sample.edge_id.shape[0] == sample.num_edges
        assert sample.edge_id.shape[0] == n_e

        # assert sample.node_norm.numel() == sample.num_nodes
        # assert sample.edge_norm.numel() == sample.num_edges


if __name__ == '__main__':
    for i in range(10):
        test_graph_saint()
