import tensorlayerx as tlx
from gammagl.layers.conv import MessagePassing
import tensorlayerx as tlx
from gammagl.mpops import *
from gammagl.utils.num_nodes import maybe_num_nodes
from gammagl.layers.conv import Hid_conv
import numpy as np
def test_hid_conv():
    num_nodes = 10
    in_channels = 8
    x = tlx.convert_to_tensor(np.random.randn(num_nodes, in_channels), dtype=tlx.float32)
    edge_index = tlx.convert_to_tensor(np.array([
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
    ]), dtype=tlx.int64)
    edge_weight = tlx.convert_to_tensor(np.ones(edge_index.shape[1]), dtype=tlx.float32)
    ei_no_loops = edge_index
    ew_no_loops = edge_weight
    alpha = 0.5
    beta = 0.3
    gamma = 0.2
    sigma1 = 0.5
    sigma2 = 0.5
    origin = x
    conv = Hid_conv(alpha, beta, gamma, sigma1, sigma2)
    out = conv(x, origin, edge_index, edge_weight, ei_no_loops, ew_no_loops)
    assert out.shape == x.shape
    print("Test passed with output shape:", out.shape)