import tensorlayerx as tlx
from gammagl.layers.conv import MessagePassing
from gammagl.utils import sort_edge_index 
from gammagl.ops.sparse import ind2ptr
from gammagl.layers.conv import FusedGATConv
import numpy as np


def test_fusedgat_conv():
    conv = FusedGATConv(in_channels=64, out_channels=128, heads=4, concat=True, dropout_rate=0.5)
    x = tlx.convert_to_tensor(np.random.randn(32, 64), dtype=tlx.float32) 
    edge_index = tlx.convert_to_tensor(np.array([[0, 1, 1, 2], [1, 0, 2, 1]]), dtype=tlx.int32)  
    output = conv(x, edge_index)
    assert isinstance(output, tlx.Tensor)
    assert output.shape == (32, 128 * 4) 
