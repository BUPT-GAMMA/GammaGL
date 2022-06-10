import tensorlayerx as tlx
from gammagl.data import Graph

from gammagl.transforms import NormalizeFeatures


def test_normalize_scale():
    assert NormalizeFeatures().__repr__() == 'NormalizeFeatures()'

    x = tlx.convert_to_tensor([[1, 0, 1], [0, 1, 0], [0, 0, 0]], dtype=tlx.float32)

    data = Graph(x=x)
    data = NormalizeFeatures()(data)
    assert len(data) == 1
    assert tlx.convert_to_numpy(data.x).tolist() == [[0.5, 0, 0.5], [0, 1, 0], [0, 0, 0]]
