import pytest
import tensorlayerx as tlx

from gammagl.data import Graph
from gammagl.transforms import RandomLinkSplit

def test_random_link_split():
    assert str(RandomLinkSplit()) == ('RandomLinkSplit('
                                      'num_val=0.1, num_test=0.2)')

    edge_index = tlx.convert_to_tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5],
                               [1, 0, 2, 1, 3, 2, 4, 3, 5, 4]])
    edge_attr = tlx.random_normal((edge_index.shape[1], 3))

    data = Graph(edge_index=edge_index, edge_attr=edge_attr, num_nodes=100)

    # No test split:
    transform = RandomLinkSplit(num_val=2, num_test=0, is_undirected=True)
    train_data, val_data, test_data = transform(data)

    assert len(train_data) == 5
    assert train_data.num_nodes == 100
    assert tuple(train_data.edge_index.shape) == (2, 6)
    assert tuple(train_data.edge_attr.shape) == (6, 3)
    assert train_data.edge_label_index.shape[1] == 6
    assert train_data.edge_label.shape[0] == 6

    assert len(val_data) == 5
    assert val_data.num_nodes == 100
    assert tuple(val_data.edge_index.shape) == (2, 6)
    assert tuple(val_data.edge_attr.shape) == (6, 3)
    assert val_data.edge_label_index.shape[1] == 4
    assert val_data.edge_label.shape[0] == 4

    assert len(test_data) == 5
    assert test_data.num_nodes == 100
    assert tuple(test_data.edge_index.shape) == (2, 10)
    assert tuple(test_data.edge_attr.shape) == (10, 3)
    assert tuple(test_data.edge_label_index.shape) == (2, 0)
    assert tuple(test_data.edge_label.shape) == (0, )

    # Percentage split:
    transform = RandomLinkSplit(num_val=0.2, num_test=0.2,
                                neg_sampling_ratio=2.0, is_undirected=False)
    train_data, val_data, test_data = transform(data)

    assert len(train_data) == 5
    assert train_data.num_nodes == 100
    assert tuple(train_data.edge_index.shape) == (2, 6)
    assert tuple(train_data.edge_attr.shape) == (6, 3)
    assert train_data.edge_label_index.shape[1] == 18
    assert train_data.edge_label.shape[0] == 18

    assert len(val_data) == 5
    assert val_data.num_nodes == 100
    assert tuple(val_data.edge_index.shape) == (2, 6)
    assert tuple(val_data.edge_attr.shape) == (6, 3)
    assert val_data.edge_label_index.shape[1] == 6
    assert val_data.edge_label.shape[0] == 6

    assert len(test_data) == 5
    assert test_data.num_nodes == 100
    assert tuple(test_data.edge_index.shape) == (2, 8)
    assert tuple(test_data.edge_attr.shape) == (8, 3)
    assert test_data.edge_label_index.shape[1] == 6
    assert test_data.edge_label.shape[0] == 6

    # Disjoint training split:
    transform = RandomLinkSplit(num_val=0.2, num_test=0.2, is_undirected=False,
                                disjoint_train_ratio=0.5)
    train_data, val_data, test_data = transform(data)

    assert len(train_data) == 5
    assert train_data.num_nodes == 100
    assert tuple(train_data.edge_index.shape) == (2, 3)
    assert tuple(train_data.edge_attr.shape) == (3, 3)
    assert train_data.edge_label_index.shape[1] == 6
    assert train_data.edge_label.shape[0] == 6