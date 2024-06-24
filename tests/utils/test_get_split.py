import tensorlayerx as tlx
from gammagl.utils import get_train_val_test_split
import numpy as np


class Graph:
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes

def test_get_split():
    num_nodes = 1000
    graph = Graph(num_nodes)

    train_ratio = 0.6
    val_ratio = 0.2

    train_mask, val_mask, test_mask = get_train_val_test_split(graph, train_ratio, val_ratio)
    
    assert tlx.ops.is_tensor(train_mask)
    assert tlx.ops.is_tensor(val_mask)
    assert tlx.ops.is_tensor(test_mask)

    train_mask = tlx.convert_to_numpy(train_mask)
    val_mask = tlx.convert_to_numpy(val_mask)
    test_mask = tlx.convert_to_numpy(test_mask)

    assert np.sum(train_mask) == int(num_nodes * train_ratio)
    assert np.sum(val_mask) == int(num_nodes * val_ratio)
    assert np.sum(test_mask) == num_nodes - int(num_nodes * train_ratio) - int(num_nodes * val_ratio)

    assert np.all(train_mask + val_mask + test_mask == 1)
    