import numpy as np
import tensorlayerx as tlx
from gammagl.datasets import Planetoid

from gammagl.loader import NeighborSampler


def test_neighbor_sampler_cora():
    dataset = Planetoid()
    graph = dataset[0]
    loader = NeighborSampler(edge_index=tlx.convert_to_numpy(graph.edge_index),
                             node_idx=np.arange(0, graph.num_nodes, dtype=np.int64)
                             , sample_lists=[-1, -1],
                             batch_size=512)

    assert len(loader) == int(1 + (graph.num_nodes / 512))

    for dst_node, n_id, adjs in loader:
        assert len(adjs) == 2
        assert int(dst_node.shape[0]) == adjs[1].size[1]
        for adj in adjs:
            assert int(tlx.reduce_max(adj.edge_index[0]) + 1) <= adj.size[0]
            assert int(tlx.reduce_max(adj.edge_index[1]) + 1) <= adj.size[1]
            assert adj.size[0] >= adj.size[1]


test_neighbor_sampler_cora()
