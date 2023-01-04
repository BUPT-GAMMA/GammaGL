import tensorlayerx as tlx
from gammagl.layers.conv import GINConv
from gammagl.models import MLP

def test_gin_conv():
    x = tlx.random_normal(shape=(4, 16))
    edge_index = tlx.convert_to_tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    mlp = MLP([16, 4, 4], norm = None)
    conv = GINConv(nn=mlp, train_eps=False)
    out = conv(x, edge_index)
    assert tlx.get_tensor_shape(out) == [4, 4]
