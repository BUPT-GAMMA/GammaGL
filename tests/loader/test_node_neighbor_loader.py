# -*- coding: utf-8 -*-
# @author WuJing
# @created 2023/3/27

import tensorlayerx as tlx
import numpy as np
from gammagl.data import HeteroGraph
from gammagl.data.graph import Graph
from gammagl.data.storage import NodeStorage, EdgeStorage
from gammagl.loader.node_neighbor_loader import NodeNeighborLoader
from gammagl.utils.platform_utils import all_to_numpy, all_to_numpy_by_dict


def randint(low, high, shape):
    return tlx.convert_to_tensor(np.random.randint(low, high, shape), dtype=tlx.int64)


def get_edge_index(num_src_nodes, num_dst_nodes, num_edges):
    row = randint(0, num_src_nodes, (num_edges,))
    col = randint(0, num_dst_nodes, (num_edges,))
    return tlx.stack([row, col], axis=0)


def is_subset(subedge_index, edge_index, src_idx, dst_idx):
    num_nodes = int(tlx.reduce_max(edge_index)) + 1
    idx = tlx.add(tlx.multiply(num_nodes * tlx.ones_like(edge_index[0]), edge_index[0]), edge_index[1])
    src = tlx.gather(src_idx, subedge_index[0])
    dst = tlx.gather(dst_idx, subedge_index[1])
    subidx = tlx.add(tlx.multiply(num_nodes * tlx.ones_like(src), src), dst)

    mask = np.isin(subidx, idx)
    return np.all(mask)


def test_homogeneous_neighbor_loader(directed=True):
    tlx.set_seed(12345)
    x = tlx.arange(start = 0, limit = 100)
    edge_index = get_edge_index(100, 100, 500)
    edge_attr = tlx.arange(start = 0, limit = 500)
    graph = Graph(x=x, edge_index=edge_index, edge_attr=edge_attr)

    loader = NodeNeighborLoader(graph,
                                num_neighbors=[5] * 2, batch_size=20,
                                directed=directed)

    assert str(loader) == 'NodeNeighborLoader()'
    assert len(loader) == 5

    for batch in loader:
        assert isinstance(batch, Graph)

        # batch = all_to_numpy(batch)

        assert batch.x.shape[0] <= 100
        assert batch.batch_size == 20
        assert tlx.reduce_min(batch.x) >= 0 and tlx.reduce_max(batch.x) < 100
        assert tlx.reduce_min(batch.edge_index) >= 0
        assert tlx.reduce_max(batch.edge_index) < batch.num_nodes
        assert tlx.reduce_min(batch.edge_attr) >= 0
        assert tlx.reduce_max(batch.edge_attr) < 500

        assert is_subset(batch.edge_index, edge_index, batch.x, batch.x)


def test_heterogeneous_neighbor_loader(directed=True):
    graph = HeteroGraph()

    graph['paper'].x = tlx.arange(start = 0, limit = 100)
    graph['author'].x = tlx.arange(start = 100, limit = 300)

    graph['paper', 'paper'].edge_index = get_edge_index(100, 100, 500)
    graph['paper', 'paper'].edge_attr = tlx.arange(start = 0, limit = 500)
    graph['paper', 'author'].edge_index = get_edge_index(100, 200, 1000)
    graph['paper', 'author'].edge_attr = tlx.arange(start = 500, limit = 1500)
    graph['author', 'paper'].edge_index = get_edge_index(200, 100, 1000)
    graph['author', 'paper'].edge_attr = tlx.arange(start = 1500, limit = 2500)

    r1, c1 = graph['paper', 'paper'].edge_index
    r2, c2 = graph['paper', 'author'].edge_index + tlx.convert_to_tensor([[0], [100]], dtype=tlx.int64)
    r3, c3 = graph['author', 'paper'].edge_index + tlx.convert_to_tensor([[100], [0]], dtype=tlx.int64)

    batch_size = 20

    # with pytest.raises(ValueError, match="to have 2 entries"):
    #     loader = NodeNeighborLoader(
    #         graph,
    #         num_neighbors={
    #             ('paper', 'paper'): [-1],
    #             ('paper', 'author'): [-1, -1],
    #             ('author', 'paper'): [-1, -1],
    #         },
    #         input_nodes='paper',
    #         batch_size=batch_size,
    #         directed=directed,
    #     )

    loader = NodeNeighborLoader(
        graph,
        num_neighbors=[10] * 2,
        input_nodes_type='paper',
        batch_size=batch_size,
        directed=directed,
    )

    assert str(loader) == 'NodeNeighborLoader()'
    assert len(loader) == (100 + batch_size - 1) // batch_size

    for batch in loader:
        assert isinstance(batch, HeteroGraph)

        # Test node type selection:
        assert set(batch.node_types) == {'paper', 'author'}
        # ['paper', 'author', ('paper', 'paper'), ('paper', 'author'), ('author', 'author')]
        # all_to_numpy_by_dict(batch, {
        #     HeteroGraph: ['paper', 'author', ('paper', 'paper'), ('paper', 'author'), ('author', 'paper')],
        #     NodeStorage: 'x',
        #     EdgeStorage: ['edge_index', 'edge_attr']
        # })

        assert batch['paper'].x.shape[0] <= 100
        assert batch['paper'].batch_size == batch_size
        assert tlx.reduce_min(batch['paper'].x) >= 0 and tlx.reduce_max(batch['paper'].x) < 100

        assert batch['author'].x.shape[0] <= 200
        assert tlx.reduce_min(batch['author'].x) >= 100 and tlx.reduce_max(batch['author'].x) < 300

        # Test edge type selection:
        assert set(batch.edge_types) == {('paper', 'to', 'paper'),
                                         ('paper', 'to', 'author'),
                                         ('author', 'to', 'paper')}

        row, col = batch['paper', 'paper'].edge_index
        value = batch['paper', 'paper'].edge_attr
        assert tlx.reduce_min(row) >= 0 and tlx.reduce_max(row) < batch['paper'].num_nodes
        assert tlx.reduce_min(col) >= 0 and tlx.reduce_max(col) < batch['paper'].num_nodes
        assert tlx.reduce_min(value) >= 0 and tlx.reduce_max(value) < 500

        assert is_subset(batch['paper', 'paper'].edge_index,
                         graph['paper', 'paper'].edge_index, batch['paper'].x,
                         batch['paper'].x)

        row, col = batch['paper', 'author'].edge_index
        value = batch['paper', 'author'].edge_attr
        assert tlx.reduce_min(row) >= 0 and tlx.reduce_max(row) < batch['paper'].num_nodes
        assert tlx.reduce_min(col) >= 0 and tlx.reduce_max(col) < batch['author'].num_nodes
        assert tlx.reduce_min(value) >= 500 and tlx.reduce_max(value) < 1500

        assert is_subset(batch['paper', 'author'].edge_index,
                         graph['paper', 'author'].edge_index, batch['paper'].x,
                         batch['author'].x - 100)

        row, col = batch['author', 'paper'].edge_index
        value = batch['author', 'paper'].edge_attr
        assert tlx.reduce_min(row) >= 0 and tlx.reduce_max(row) < batch['author'].num_nodes
        assert tlx.reduce_min(col) >= 0 and tlx.reduce_max(col) < batch['paper'].num_nodes
        assert tlx.reduce_min(value) >= 1500 and tlx.reduce_max(value) < 2500

        assert is_subset(batch['author', 'paper'].edge_index,
                         graph['author', 'paper'].edge_index,
                         batch['author'].x - 100, batch['paper'].x)

test_homogeneous_neighbor_loader()
test_heterogeneous_neighbor_loader()