import tensorlayerx as tlx
from gammagl.datasets import ACM4DHN

root = "./data"


def test_acm4dhn():
    test_ratio = 0.3
    dataset = ACM4DHN(root=root, test_ratio=test_ratio)
    graph = dataset[0]
    assert len(dataset) == 1
    assert tlx.get_tensor_shape(graph['M', 'MA', 'A'].edge_index) == [2, 7071]
    assert tlx.get_tensor_shape(graph['train']['M', 'MA', 'A'].edge_index) == [2, 2828]
    assert tlx.get_tensor_shape(graph['val']['M', 'MA', 'A'].edge_index) == [2, 2121]
    assert tlx.get_tensor_shape(graph['test']['M', 'MA', 'A'].edge_index) == [2, 2122]
