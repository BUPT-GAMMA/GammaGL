import tensorlayerx as tlx
from tensorlayerx.nn import Linear as Lin
from tensorlayerx.nn import ReLU
from tensorlayerx.nn import Sequential as Seq

from gammagl.layers.conv import EdgeConv

def test_edge_conv_conv():
    x1 = tlx.random_normal(shape = (4, 16))
    edge_index = tlx.convert_to_tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    nn = Seq(Lin(in_features = 32, out_features = 16), ReLU(), Lin(in_features = 16, out_features = 32))
    conv = EdgeConv(nn)

    out1 = conv(x1, edge_index)
    assert tlx.get_tensor_shape(out1) == [4, 32]
    assert tlx.convert_to_numpy(conv((x1, x1), edge_index)).tolist() == tlx.convert_to_numpy(out1).tolist()
