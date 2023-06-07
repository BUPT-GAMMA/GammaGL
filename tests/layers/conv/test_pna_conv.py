import tensorlayerx as tlx
from gammagl.layers.conv import PNAConv

aggregators = ['sum', 'mean', 'min', 'max', 'var', 'std']
scalers = [
    'identity', 'amplification', 'attenuation', 'linear', 'inverse_linear'
]


def test_pna_conv():
    x = tlx.random_normal((4, 16))
    edge_index = tlx.convert_to_tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])

    conv = PNAConv(16, 32, aggregators, scalers,
                   deg=tlx.convert_to_tensor([0, 3, 0, 1]), edge_dim=None, towers=4)
    out = conv(x, edge_index)
    assert tlx.get_tensor_shape(out) == [4, 32]
