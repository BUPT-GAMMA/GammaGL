# -*- coding: utf-8 -*-
# @author WuJing
# @created 2023/3/23

import numpy as np
import tensorlayerx as tlx
from gammagl.data.graph import Graph
from gammagl.loader.link_neighbor_loader import LinkNeighborLoader


def randint(low, high, shape):
    return tlx.convert_to_tensor(np.random.randint(low, high, shape), dtype=tlx.int64)


def get_edge_index(num_src_nodes, num_dst_nodes, num_edges, is_neg=False):
    row = randint(0, num_src_nodes, (num_edges,))
    if not is_neg:
        col = randint(0, num_dst_nodes, (num_edges,))
    else:
        col = randint(num_dst_nodes, 2 * num_dst_nodes, (num_edges,))
    return tlx.stack([row, col], axis=0)


def unique_edge_pairs(edge_index):
    return set(map(tuple, tlx.convert_to_numpy(tlx.transpose(edge_index)).tolist()))


def homogeneous_link_neighbor_loader(directed, neg_sampling_ratio):
    pos_edge_index = get_edge_index(100, 50, 500)  # 2,500
    neg_edge_index = get_edge_index(100, 50, 500)  # 2,500

    edge_label_index = tlx.concat([pos_edge_index, neg_edge_index], axis=-1)  # 2,1000
    edge_label = tlx.concat([tlx.ones(500), tlx.zeros(500)], axis=0)  # 1000

    graph = Graph(x=tlx.arange(100, dtype=tlx.int64), edge_index=pos_edge_index, edge_attr=tlx.arange(500))

    # if set a neg_sampling_ratio, the original edge_label will be invalid,
    #       and a new edge_label will be in the process of sampling.
    # positive label is one, negative label is zero.
    loader = LinkNeighborLoader(
        graph,
        num_neighbors=[-1] * 2,
        batch_size=20,
        edge_label_index=edge_label_index,
        edge_label=edge_label if neg_sampling_ratio == 0.0 else None,
        directed=directed,
        neg_sampling_ratio=neg_sampling_ratio,
        shuffle=False,
    )

    assert str(loader) == 'LinkNeighborLoader()'
    assert len(loader) == 1000 / 20

    for batch in loader:
        assert isinstance(batch, Graph)

        assert len(batch) == 6
        assert batch.x.shape[0] <= 100
        assert tlx.reduce_min(batch.x) >= 0 and tlx.reduce_max(batch.x) < 100
        assert tlx.reduce_min(batch.edge_index) >= 0
        assert tlx.reduce_max(batch.edge_index) < batch.num_nodes
        assert tlx.reduce_min(batch.edge_attr) >= 0
        assert tlx.reduce_max(batch.edge_attr) < 500

        batch.edge_index = tlx.convert_to_numpy(batch.edge_index)
        batch.edge_label = tlx.convert_to_numpy(batch.edge_label)
        batch.edge_label_index = tlx.convert_to_numpy(batch.edge_label_index)

        if neg_sampling_ratio == 0.0:
            assert batch.edge_label_index.shape[1] == 20

            # Assert positive samples are present in the original graph:
            edge_index = unique_edge_pairs(batch.edge_index)
            edge_label_index = batch.edge_label_index[:, batch.edge_label == 1]
            edge_label_index = unique_edge_pairs(edge_label_index)
            assert len(edge_index | edge_label_index) == len(edge_index)

            # Assert negative samples are not present in the original graph:
            edge_index = unique_edge_pairs(batch.edge_index)
            edge_label_index = batch.edge_label_index[:, batch.edge_label == 0]
            edge_label_index = unique_edge_pairs(edge_label_index)
            assert len(edge_index & edge_label_index) == 0

        else:
            assert batch.edge_label_index.shape[1] == 40
            assert tlx.all(batch.edge_label[:20] == 1)
            assert tlx.all(batch.edge_label[20:] == 0)


