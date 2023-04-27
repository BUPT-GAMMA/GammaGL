import tensorlayerx as tlx
from gammagl.layers.conv import GMMConv
from gammagl.utils import add_self_loops


def test_GMMConv_conv():
    x = tlx.random_normal((7, 10))
    edge_index = tlx.convert_to_tensor([[0, 1, 2, 3, 6, 2, 5],
                                        [1, 2, 3, 4, 2, 0, 3]], dtype=tlx.int64)
    edge_index, _ = add_self_loops(edge_index)
    pseudo = tlx.ones((14, 2))
    conv = GMMConv(10, 2, 2, 3, 'mean')
    res = conv(x, edge_index, pseudo)
    assert tlx.get_tensor_shape(res) == [7, 2]