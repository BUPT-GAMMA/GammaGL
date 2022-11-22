import tensorlayerx as tlx

from gammagl.models.mlp import MLP

tlx.set_device('GPU', 4)


def test_mlp():
    mlp = MLP([1, 5, 10])
    tensor = tlx.convert_to_tensor([[1], [2], [3]], dtype=tlx.float32)
    out = mlp(tensor)
    assert out.shape == (3, 10)


test_mlp()
