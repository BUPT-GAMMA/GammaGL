# -*- coding: utf-8 -*-
# @author WuJing
# @created 2023/4/1

import numpy as np
from gammagl.data.graph import Graph
from gammagl.utils.random_walk_sample import rw_sample, rw_sample_by_edge_index


def randint(low, high, shape):
    return np.random.randint(low, high, shape)


def get_edge_index(num_src_nodes, num_dst_nodes, num_edges):
    row = randint(0, num_src_nodes, (num_edges,))
    col = randint(0, num_dst_nodes, (num_edges,))
    return np.stack([row, col], axis=0)


def test_random_walk():
    # tlx.set_seed(12345)
    x = np.arange(100, dtype=np.int32)

    edge_index = get_edge_index(100, 100, 500)
    edge_attr = np.arange(500)
    graph = Graph(x=x, edge_index=edge_index, edge_attr=edge_attr)

    result = rw_sample(graph, x[:20], 3)
    assert isinstance(result, list)
    result = rw_sample_by_edge_index(graph.edge_index, x[:20], 3)
    assert isinstance(result, list)
