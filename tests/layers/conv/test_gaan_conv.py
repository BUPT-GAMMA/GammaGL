import tensorlayerx as tlx
from gammagl.layers.conv import MessagePassing
from gammagl.mpops import *
from gammagl.utils import segment_softmax
from gammagl.layers.conv import GaANConv
import tensorlayerx as tlx
import numpy as np
from gammagl.utils import to_undirected

def test_gaan_conv():
    num_nodes = 10
    in_channels = 8
    out_channels = 16
    heads = 4

    x = tlx.convert_to_tensor(np.random.randn(num_nodes, in_channels), dtype=tlx.float32)
    edge_index = tlx.convert_to_tensor(np.array([
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
    ]), dtype=tlx.int32)

    edge_index = to_undirected(edge_index)

    conv = GaANConv(in_channels=in_channels, out_channels=out_channels, heads=heads)
    out = conv(x, edge_index)

    assert out.shape == (num_nodes, out_channels * heads)
    


