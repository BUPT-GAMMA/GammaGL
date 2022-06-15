import tensorlayerx as tlx
from gammagl.data import Graph
import gammagl.transforms as T


def test_compose():
    transform = T.Compose([T.NormalizeFeatures(), T.SIGN(2)])
    assert str(transform) == 'Compose([\n  NormalizeFeatures(),\n  SIGN(K=2)\n])'

    x = tlx.convert_to_tensor([[2, 6], [2, 0], [4, 9]])
    edge_index = tlx.convert_to_tensor([[0, 1, 1, 2], [1, 0, 2, 1]])

    data = Graph(edge_index=edge_index, x=x)
    data = transform(data)
    assert len(data) == 4
    assert tlx.get_tensor_shape(data.edge_index) == [2, 4]
    assert tlx.all(data.x == tlx.convert_to_tensor([[0.25, 0.75],[1., 0.],[0.30769232, 0.6923077]], dtype=tlx.float32))
    for i in range(1, 3):
        assert data[f'x{i}'].shape == (3, 2)
