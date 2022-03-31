import copy
import tensorlayerx as tlx
from gammagl.data import Graph

def test_data():

    x = tlx.ops.constant([[1, 3, 5], [2, 4, 6]], dtype=tlx.float32)
    edge_index = tlx.ops.constant([[0, 0, 1, 1, 2], [1, 1, 0, 2, 1]])
    data = Graph(x=x, edge_index=edge_index)

    N = data.num_nodes
    assert N == 3
