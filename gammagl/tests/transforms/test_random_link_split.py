import pytest
import tensorlayerx as tlx
import numpy as np

from gammagl.data import Graph, HeteroGraph
from gammagl.transforms import RandomLinkSplit
from gammagl.utils import to_undirected, is_undirected

def test_random_link_split():
    assert str(RandomLinkSplit()) == ('RandomLinkSplit('
                                      'num_val=0.1, num_test=0.2)')

    edge_index = tlx.convert_to_tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5],
                               [1, 0, 2, 1, 3, 2, 4, 3, 5, 4]], dtype = tlx.int64)
    edge_attr = tlx.random_normal((tlx.get_tensor_shape(edge_index)[1], 3), dtype = tlx.float32)

    data = Graph(edge_index=edge_index, edge_attr=edge_attr, num_nodes=100)

    # No test split:
    transform = RandomLinkSplit(num_val=2, num_test=0, is_undirected=True)
    train_data, val_data, test_data = transform(data)

    assert len(train_data) == 5
    assert train_data.num_nodes == 100
    assert tlx.get_tensor_shape(train_data.edge_index) == [2, 6]
    assert tlx.get_tensor_shape(train_data.edge_attr) == [6, 3]
    assert tlx.get_tensor_shape(train_data.edge_label_index)[1] == 6
    assert tlx.get_tensor_shape(train_data.edge_label)[0] == 6

    assert len(val_data) == 5
    assert val_data.num_nodes == 100
    assert tlx.get_tensor_shape(val_data.edge_index) == [2, 6]
    assert tlx.get_tensor_shape(val_data.edge_attr) == [6, 3]
    assert tlx.get_tensor_shape(val_data.edge_label_index)[1] == 4
    assert tlx.get_tensor_shape(val_data.edge_label)[0] == 4

    assert len(test_data) == 5
    assert test_data.num_nodes == 100
    assert tlx.get_tensor_shape(test_data.edge_index) == [2, 10]
    assert tlx.get_tensor_shape(test_data.edge_attr) == [10, 3]
    assert tlx.get_tensor_shape(test_data.edge_label_index) == [2, 0]
    assert tlx.get_tensor_shape(test_data.edge_label) == [0, ]

    # Percentage split:
    transform = RandomLinkSplit(num_val=0.2, num_test=0.2,
                                neg_sampling_ratio=2.0, is_undirected=False)
    train_data, val_data, test_data = transform(data)

    assert len(train_data) == 5
    assert train_data.num_nodes == 100
    assert tlx.get_tensor_shape(train_data.edge_index) == [2, 6]
    assert tlx.get_tensor_shape(train_data.edge_attr) == [6, 3]
    assert tlx.get_tensor_shape(train_data.edge_label_index)[1] == 18
    assert tlx.get_tensor_shape(train_data.edge_label)[0] == 18

    assert len(val_data) == 5
    assert val_data.num_nodes == 100
    assert tlx.get_tensor_shape(val_data.edge_index) == [2, 6]
    assert tlx.get_tensor_shape(val_data.edge_attr) == [6, 3]
    assert tlx.get_tensor_shape(val_data.edge_label_index)[1] == 6
    assert tlx.get_tensor_shape(val_data.edge_label)[0] == 6

    assert len(test_data) == 5
    assert test_data.num_nodes == 100
    assert tlx.get_tensor_shape(test_data.edge_index) == [2, 8]
    assert tlx.get_tensor_shape(test_data.edge_attr) == [8, 3]
    assert tlx.get_tensor_shape(test_data.edge_label_index)[1] == 6
    assert tlx.get_tensor_shape(test_data.edge_label)[0] == 6

    # Disjoint training split:
    transform = RandomLinkSplit(num_val=0.2, num_test=0.2, is_undirected=False,
                                disjoint_train_ratio=0.5)
    train_data, val_data, test_data = transform(data)

    assert len(train_data) == 5
    assert train_data.num_nodes == 100
    assert tlx.get_tensor_shape(train_data.edge_index) == [2, 3]
    assert tlx.get_tensor_shape(train_data.edge_attr) == [3, 3]
    assert tlx.get_tensor_shape(train_data.edge_label_index)[1] == 6
    assert tlx.get_tensor_shape(train_data.edge_label)[0] == 6

def test_random_link_split_on_hetero_data():
    data = HeteroGraph()

    data['p'].x = tlx.arange(0, 100, dtype = tlx.float32)
    data['a'].x = tlx.arange(100, 300, dtype = tlx.float32)

    row = tlx.convert_to_tensor(np.random.randint(0, 100, size = (500, )), dtype = tlx.int64)
    col = tlx.convert_to_tensor(np.random.randint(0, 100, size = (500, )), dtype = tlx.int64)
    data['p', 'p'].edge_index = tlx.stack([row, col])
    data['p', 'p'].edge_index = to_undirected(data['p', 'p'].edge_index)
    data['p', 'p'].edge_attr = tlx.arange(0, data['p', 'p'].num_edges, dtype = tlx.float32)
    row = tlx.convert_to_tensor(np.random.randint(0, 100, size = (1000, )), dtype = tlx.int64)
    col = tlx.convert_to_tensor(np.random.randint(0, 200, size = (1000, )), dtype = tlx.int64)
    data['p', 'a'].edge_index = tlx.stack([row, col])
    data['p', 'a'].edge_attr = tlx.arange(500, 1500, dtype = tlx.float32)
    data['a', 'p'].edge_index = tlx.gather(data['p', 'a'].edge_index, tlx.convert_to_tensor([1, 0]))
    data['a', 'p'].edge_attr = tlx.arange(1500, 2500, dtype = tlx.float32)

    transform = RandomLinkSplit(num_val=0.2, num_test=0.2, is_undirected=True,
                                edge_types=('p', 'p'))
    train_data, val_data, test_data = transform(data)

    assert len(train_data['p']) == 1
    assert len(train_data['a']) == 1
    assert len(train_data['p', 'p']) == 4
    assert len(train_data['p', 'a']) == 2
    assert len(train_data['a', 'p']) == 2

    assert is_undirected(train_data['p', 'p'].edge_index,
                         train_data['p', 'p'].edge_attr)
    assert is_undirected(val_data['p', 'p'].edge_index,
                         val_data['p', 'p'].edge_attr)
    assert is_undirected(test_data['p', 'p'].edge_index,
                         test_data['p', 'p'].edge_attr)

    transform = RandomLinkSplit(num_val=0.2, num_test=0.2,
                                edge_types=('p', 'a'),
                                rev_edge_types=('a', 'p'))
    train_data, val_data, test_data = transform(data)

    assert len(train_data['p']) == 1
    assert len(train_data['a']) == 1
    assert len(train_data['p', 'p']) == 2
    assert len(train_data['p', 'a']) == 4
    assert len(train_data['a', 'p']) == 2

    assert tlx.get_tensor_shape(train_data['p', 'a'].edge_index) == [2, 600]
    assert tlx.get_tensor_shape(train_data['p', 'a'].edge_attr) == [600, ]
    assert min(train_data['p', 'a'].edge_attr) >= 500
    assert max(train_data['p', 'a'].edge_attr) <= 1500
    assert tlx.get_tensor_shape(train_data['a', 'p'].edge_index) == [2, 600]
    assert tlx.get_tensor_shape(train_data['a', 'p'].edge_attr) == [600, ]
    assert min(train_data['a', 'p'].edge_attr) >= 500
    assert max(train_data['a', 'p'].edge_attr) <= 1500
    assert tlx.get_tensor_shape(train_data['p', 'a'].edge_label_index) == [2, 1200]
    assert tlx.get_tensor_shape(train_data['p', 'a'].edge_label) == [1200, ]

    assert tlx.get_tensor_shape(val_data['p', 'a'].edge_index) == [2, 600]
    assert tlx.get_tensor_shape(val_data['p', 'a'].edge_attr) == [600, ]
    assert min(val_data['p', 'a'].edge_attr) >= 500
    assert max(val_data['p', 'a'].edge_attr) <= 1500
    assert tlx.get_tensor_shape(val_data['a', 'p'].edge_index) == [2, 600]
    assert tlx.get_tensor_shape(val_data['a', 'p'].edge_attr) == [600, ]
    assert min(val_data['a', 'p'].edge_attr) >= 500
    assert max(val_data['a', 'p'].edge_attr) <= 1500
    assert tlx.get_tensor_shape(val_data['p', 'a'].edge_label_index) == [2, 400]
    assert tlx.get_tensor_shape(val_data['p', 'a'].edge_label) == [400, ]

    assert tlx.get_tensor_shape(test_data['p', 'a'].edge_index) == [2, 800]
    assert tlx.get_tensor_shape(test_data['p', 'a'].edge_attr) == [800, ]
    assert min(test_data['p', 'a'].edge_attr) >= 500
    assert max(test_data['p', 'a'].edge_attr) <= 1500
    assert tlx.get_tensor_shape(test_data['a', 'p'].edge_index) == [2, 800]
    assert tlx.get_tensor_shape(test_data['a', 'p'].edge_attr) == [800, ]
    assert min(test_data['a', 'p'].edge_attr) >= 500
    assert max(test_data['a', 'p'].edge_attr) <= 1500
    assert tlx.get_tensor_shape(test_data['p', 'a'].edge_label_index) == [2, 400]
    assert tlx.get_tensor_shape(test_data['p', 'a'].edge_label) == [400, ]

    transform = RandomLinkSplit(num_val=0.2, num_test=0.2, is_undirected=True,
                                edge_types=[('p', 'p'), ('p', 'a')],
                                rev_edge_types=[None, ('a', 'p')])
    train_data, val_data, test_data = transform(data)

    assert len(train_data['p']) == 1
    assert len(train_data['a']) == 1
    assert len(train_data['p', 'p']) == 4
    assert len(train_data['p', 'a']) == 4
    assert len(train_data['a', 'p']) == 2

    assert is_undirected(train_data['p', 'p'].edge_index,
                         train_data['p', 'p'].edge_attr)
    assert tlx.get_tensor_shape(train_data['p', 'a'].edge_index) == [2, 600]
    assert tlx.get_tensor_shape(train_data['a', 'p'].edge_index) == [2, 600]


def test_random_link_split_on_undirected_hetero_data():
    data = HeteroGraph()
    data['p'].x = tlx.arange(0, 100, dtype = tlx.float32)
    row = tlx.convert_to_tensor(np.random.randint(0, 100, size = (500, )), dtype = tlx.int64)
    col = tlx.convert_to_tensor(np.random.randint(0, 100, size = (500, )), dtype = tlx.int64)
    data['p', 'p'].edge_index = tlx.stack([row, col])
    data['p', 'p'].edge_index = to_undirected(data['p', 'p'].edge_index)

    transform = RandomLinkSplit(is_undirected=True, edge_types=('p', 'p'))
    train_data, val_data, test_data = transform(data)
    assert train_data['p', 'p'].is_undirected()

    transform = RandomLinkSplit(is_undirected=True, edge_types=('p', 'p'),
                                rev_edge_types=('p', 'p'))
    train_data, val_data, test_data = transform(data)
    assert train_data['p', 'p'].is_undirected()

    transform = RandomLinkSplit(is_undirected=True, edge_types=('p', 'p'),
                                rev_edge_types=('p', 'p'))
    train_data, val_data, test_data = transform(data)
    assert train_data['p', 'p'].is_undirected()


def test_random_link_split_insufficient_negative_edges():
    edge_index = tlx.convert_to_tensor([[0, 0, 1, 1, 2, 2], [1, 3, 0, 2, 0, 1]], dtype = tlx.int64)
    data = Graph(edge_index=edge_index, num_nodes=4)

    transform = RandomLinkSplit(num_val=0.34, num_test=0.34,
                                is_undirected=False, neg_sampling_ratio=2,
                                split_labels=True)

    # with pytest.warns(UserWarning, match="not enough negative edges"):
    train_data, val_data, test_data = transform(data)
    assert tlx.get_tensor_shape(train_data.neg_edge_label_index) == [2, 2]
    assert tlx.get_tensor_shape(val_data.neg_edge_label_index) == [2, 2]
    assert tlx.get_tensor_shape(test_data.neg_edge_label_index) == [2, 2]
