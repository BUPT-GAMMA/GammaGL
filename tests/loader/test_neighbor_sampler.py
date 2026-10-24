import numpy as np
import tensorlayerx as tlx
from gammagl.data import Graph

from gammagl.loader import NeighborSampler


def test_neighbor_sampler():
    # Bidirectional chain: exercise native two-hop sampling without downloads.
    graph = Graph(edge_index=tlx.convert_to_tensor([
        [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6],
        [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5],
    ], dtype=tlx.int64), num_nodes=7)
    loader = NeighborSampler(edge_index=graph.edge_index,
                             node_idx=tlx.arange(start = 0, limit = graph.num_nodes, dtype=tlx.int64)
                             , sample_lists=[-1, -1],
                             batch_size=3)

    assert len(loader) == 3

    for dst_node, n_id, adjs in loader:
        assert len(adjs) == 2
        assert int(dst_node.shape[0]) == adjs[1].size[1]
        for adj in adjs:
            assert int(tlx.reduce_max(adj.edge_index[0]) + 1) <= adj.size[0]
            assert int(tlx.reduce_max(adj.edge_index[1]) + 1) <= adj.size[1]
            assert adj.size[0] >= adj.size[1]
