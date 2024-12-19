import tensorlayerx as tlx
from gammagl.layers.conv import HardGATConv

def test_hardgat_conv():
    x = tlx.random_normal(shape=(4, 16))
    edge_index = tlx.convert_to_tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    conv = HardGATConv(16, 4)
    out = conv(x, edge_index, 4)
    assert tlx.get_tensor_shape(out) == [4, 4]
