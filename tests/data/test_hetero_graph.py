import os
os.environ['TL_BACKEND'] = 'tensorflow'
import pytest
import copy
import tensorlayerx as tlx
import numpy as np
from gammagl.data import HeteroGraph
from gammagl.data.storage import EdgeStorage

x_paper = tlx.convert_to_tensor(np.random.randn(10, 16))
x_author = tlx.convert_to_tensor(np.random.randn(5, 32))
x_conference = tlx.convert_to_tensor(np.random.randn(5, 8))

idx_paper = tlx.convert_to_tensor(np.random.randint(tlx.get_tensor_shape(x_paper)[0], size=(100, ), dtype=np.int64))
idx_author = tlx.convert_to_tensor(np.random.randint(tlx.get_tensor_shape(x_author)[0], size=(100, ), dtype=np.int64))
idx_conference = tlx.convert_to_tensor(np.random.randint(tlx.get_tensor_shape(x_conference)[0], size=(100, ), dtype=np.int64))

edge_index_paper_paper = tlx.stack([idx_paper[:50], idx_paper[:50]], axis=0)
edge_index_paper_author = tlx.stack([idx_paper[:30], idx_author[:30]], axis=0)
edge_index_author_paper = tlx.stack([idx_author[:30], idx_paper[:30]], axis=0)
edge_index_paper_conference = tlx.stack(
    [idx_paper[:25], idx_conference[:25]], axis=0)

edge_attr_paper_paper = tlx.convert_to_tensor(np.random.randn(tlx.get_tensor_shape(edge_index_paper_paper)[1], 8))
edge_attr_author_paper = tlx.convert_to_tensor(np.random.randn(tlx.get_tensor_shape(edge_index_author_paper)[1], 8))


def get_edge_index(num_src_nodes, num_dst_nodes, num_edges):
    row = tlx.ops.convert_to_tensor(np.random.randint(num_src_nodes, size=(num_edges, ), dtype=np.int64))
    col = tlx.ops.convert_to_tensor(np.random.randint(num_dst_nodes, size=(num_edges, ), dtype=np.int64))
    return tlx.stack([row, col], axis=0)


def test_init_hetero_data():
    data = HeteroGraph()
    data['v1'].x = 1
    data['paper'].x = x_paper
    data['author'].x = x_author
    data['paper', 'paper'].edge_index = edge_index_paper_paper
    data['paper', 'author'].edge_index = edge_index_paper_author
    data['author', 'paper'].edge_index = edge_index_author_paper

    assert len(data) == 2
    assert data.node_types == ['v1', 'paper', 'author']
    assert len(data.node_stores) == 3
    assert len(data.node_items()) == 3
    assert len(data.edge_types) == 3
    assert len(data.edge_stores) == 3
    assert len(data.edge_items()) == 3

    data = HeteroGraph(
        v1={'x': 1},
        paper={'x': x_paper},
        author={'x': x_author},
        paper__paper={'edge_index': edge_index_paper_paper},
        paper__author={'edge_index': edge_index_paper_author},
        author__paper={'edge_index': edge_index_author_paper},
    )

    assert len(data) == 2
    assert data.node_types == ['v1', 'paper', 'author']
    assert len(data.node_stores) == 3
    assert len(data.node_items()) == 3
    assert len(data.edge_types) == 3
    assert len(data.edge_stores) == 3
    assert len(data.edge_items()) == 3

    data = HeteroGraph({
        'v1': {
            'x': 1
        },
        'paper': {
            'x': x_paper
        },
        'author': {
            'x': x_author
        },
        ('paper', 'paper'): {
            'edge_index': edge_index_paper_paper
        },
        ('paper', 'author'): {
            'edge_index': edge_index_paper_author
        },
        ('author', 'paper'): {
            'edge_index': edge_index_author_paper
        },
    })

    assert len(data) == 2
    assert data.node_types == ['v1', 'paper', 'author']
    assert len(data.node_stores) == 3
    assert len(data.node_items()) == 3
    assert len(data.edge_types) == 3
    assert len(data.edge_stores) == 3
    assert len(data.edge_items()) == 3


def test_hetero_data_functions():
    data = HeteroGraph()
    data['paper'].x = x_paper
    data['author'].x = x_author
    data['paper', 'paper'].edge_index = edge_index_paper_paper
    data['paper', 'author'].edge_index = edge_index_paper_author
    data['author', 'paper'].edge_index = edge_index_author_paper
    data['paper', 'paper'].edge_attr = edge_attr_paper_paper
    assert len(data) == 3
    assert sorted(data.keys) == ['edge_attr', 'edge_index', 'x']
    assert 'x' in data and 'edge_index' in data and 'edge_attr' in data
    assert data.num_nodes == 15
    assert data.num_edges == 110

    assert data.num_node_features == {'paper': 16, 'author': 32}
    assert data.num_edge_features == {
        ('paper', 'to', 'paper'): 8,
        ('paper', 'to', 'author'): 0,
        ('author', 'to', 'paper'): 0,
    }

    node_types, edge_types = data.metadata()
    assert node_types == ['paper', 'author']
    assert edge_types == [
        ('paper', 'to', 'paper'),
        ('paper', 'to', 'author'),
        ('author', 'to', 'paper'),
    ]

    x_dict = data.collect('x')
    assert len(x_dict) == 2
    assert tlx.convert_to_numpy(x_dict['paper']).tolist() == tlx.convert_to_numpy(x_paper).tolist()
    assert tlx.convert_to_numpy(x_dict['author']).tolist() == tlx.convert_to_numpy(x_author).tolist()
    assert x_dict == data.x_dict

    data.y = 0
    assert data['y'] == 0 and data.y == 0
    assert len(data) == 4
    assert sorted(data.keys) == ['edge_attr', 'edge_index', 'x', 'y']

    del data['paper', 'author']
    node_types, edge_types = data.metadata()
    assert node_types == ['paper', 'author']
    assert edge_types == [('paper', 'to', 'paper'), ('author', 'to', 'paper')]

    assert len(data.to_dict()) == 5
    assert len(data.to_namedtuple()) == 5
    assert data.to_namedtuple().y == 0
    assert len(data.to_namedtuple().paper) == 1


def test_hetero_data_rename():
    data = HeteroGraph()
    data['paper'].x = x_paper
    data['author'].x = x_author
    data['paper', 'paper'].edge_index = edge_index_paper_paper
    data['paper', 'author'].edge_index = edge_index_paper_author
    data['author', 'paper'].edge_index = edge_index_author_paper

    data = data.rename('paper', 'article')
    assert data.node_types == ['author', 'article']
    assert data.edge_types == [
        ('article', 'to', 'article'),
        ('article', 'to', 'author'),
        ('author', 'to', 'article'),
    ]

    assert tlx.convert_to_numpy(data['article'].x).tolist() == tlx.convert_to_numpy(x_paper).tolist()
    edge_index = data['article', 'article'].edge_index
    assert tlx.convert_to_numpy(edge_index).tolist() == tlx.convert_to_numpy(edge_index_paper_paper).tolist()




def test_copy_hetero_data():
    data = HeteroGraph()
    data['paper'].x = x_paper
    data['paper', 'to', 'paper'].edge_index = edge_index_paper_paper

    out = copy.copy(data)
    assert id(data) != id(out)
    assert len(data.stores) == len(out.stores)
    for store1, store2 in zip(data.stores, out.stores):
        assert id(store1) != id(store2)
        assert id(data) == id(store1._parent())
        assert id(out) == id(store2._parent())
    assert out['paper']._key == 'paper'
    # assert data['paper'].x.data_ptr() == out['paper'].x.data_ptr()
    assert out['to']._key == ('paper', 'to', 'paper')
    # assert data['to'].edge_index.data_ptr() == out['to'].edge_index.data_ptr()

    out = copy.deepcopy(data)
    assert id(data) != id(out)
    assert len(data.stores) == len(out.stores)
    for store1, store2 in zip(data.stores, out.stores):
        assert id(store1) != id(store2)
    assert id(out) == id(out['paper']._parent())
    assert out['paper']._key == 'paper'
    # assert data['paper'].x.data_ptr() != out['paper'].x.data_ptr()
    assert tlx.convert_to_numpy(data['paper'].x).tolist() == tlx.convert_to_numpy(out['paper'].x).tolist()
    assert id(out) == id(out['to']._parent())
    assert out['to']._key == ('paper', 'to', 'paper')
    # assert data['to'].edge_index.data_ptr() != out['to'].edge_index.data_ptr()
    assert tlx.convert_to_numpy(data['to'].edge_index).tolist() == tlx.convert_to_numpy(out['to'].edge_index).tolist()


def test_hetero_data_to_canonical():
    data = HeteroGraph()
    assert isinstance(data['user', 'product'], EdgeStorage)
    assert len(data.edge_types) == 1
    assert isinstance(data['user', 'to', 'product'], EdgeStorage)
    assert len(data.edge_types) == 1

    data = HeteroGraph()
    assert isinstance(data['user', 'buys', 'product'], EdgeStorage)
    assert isinstance(data['user', 'clicks', 'product'], EdgeStorage)
    assert len(data.edge_types) == 2

    with pytest.raises(TypeError, match="missing 1 required"):
        data['user', 'product']

