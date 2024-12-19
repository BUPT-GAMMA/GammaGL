import tensorlayerx as tlx

from gammagl.data import Graph, HeteroGraph
from gammagl.transforms import SVDFeatureReduction


def test_normalize_scale():
    assert SVDFeatureReduction(2).__repr__() == 'SVDFeatureReduction()'

    x = tlx.convert_to_tensor([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]], dtype=tlx.float32)

    data = Graph(x=x)
    data = SVDFeatureReduction(out_channels=2)(data)
    assert len(data) == 1
    assert tlx.get_tensor_shape(data.x) == [2, 2]
