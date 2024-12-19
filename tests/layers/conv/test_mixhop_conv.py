import tensorlayerx as tlx
from gammagl.layers.conv import MixHopConv

def test_mixhop_conv():
    x = tlx.random_normal(shape=(4, 16))
    edge_index = tlx.convert_to_tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    conv = MixHopConv(16, 4, [0], norm='none')
    out = conv(x, edge_index)
    assert tlx.get_tensor_shape(out) == [4, 4]
