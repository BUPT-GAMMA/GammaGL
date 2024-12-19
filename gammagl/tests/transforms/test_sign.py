import tensorlayerx as tlx
from gammagl.data import Graph
from gammagl.transforms import SIGN


def test_sign():
    edge_index = tlx.convert_to_tensor([[0, 1, 2, 0, 3, 1, 0, 2, 3], [1, 0, 0, 2, 0, 0, 2, 3, 3]], dtype=tlx.int64)
    x = tlx.convert_to_tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]], dtype=tlx.float32)
    graph = Graph(edge_index=edge_index, x=x)

    K = 2
    sign = SIGN(K=K)
    graph = sign(graph)

    assert 'x1' in graph
    assert 'x2' in graph
    assert graph['x1'].shape == x.shape
    assert graph['x2'].shape == x.shape

    print("All tests passed!")
