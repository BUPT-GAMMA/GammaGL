import tensorlayerx as tlx

from gammagl.data import Graph, HeteroGraph
from gammagl.transforms import NormalizeFeatures


def test_normalize_scale():
    assert NormalizeFeatures().__repr__() == 'NormalizeFeatures()'

    x = tlx.convert_to_tensor([[1, 0, 1], [0, 1, 0], [0, 0, 0]], dtype=tlx.float32)

    data = Graph(x=x)
    data = NormalizeFeatures()(data)
    assert len(data) == 1
    assert tlx.ops.convert_to_numpy(data.x).tolist() == [[0.5, 0, 0.5], [0, 1, 0], [0, 0, 0]]


def test_hetero_normalize_scale():
    x = tlx.convert_to_tensor([[1, 0, 1], [0, 1, 0], [0, 0, 0]], dtype=tlx.float32)

    data = HeteroGraph()
    data['v'].x = x
    data['w'].x = x
    data = NormalizeFeatures()(data)
    assert tlx.ops.convert_to_numpy(data['v'].x).tolist() == [[0.5, 0, 0.5], [0, 1, 0], [0, 0, 0]]
    assert tlx.ops.convert_to_numpy(data['w'].x).tolist() == [[0.5, 0, 0.5], [0, 1, 0], [0, 0, 0]]
