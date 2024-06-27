import math
import tensorlayerx as tlx
from gammagl.utils import degree
from gammagl.layers.conv import MessagePassing
from gammagl.layers.conv import MAGCLConv
import numpy as np


def test_magcl_conv():
    in_channels = 16
    out_channels = 8
    num_nodes = 10
    edge_index = tlx.convert_to_tensor(np.array([
        [0, 1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5, 6]
    ]), dtype=tlx.int64)
    x = tlx.convert_to_tensor(np.random.randn(num_nodes, in_channels), dtype=tlx.float32)
    edge_weight = tlx.convert_to_tensor(np.random.rand(edge_index.shape[1], 1), dtype=tlx.float32)

    conv = MAGCLConv(in_channels=in_channels, out_channels=out_channels)

    try:
        out = conv(x, edge_index=edge_index, k=2, edge_weight=edge_weight, num_nodes=num_nodes)
        assert out.shape == (num_nodes, out_channels)
    except Exception as e:
        assert False, f"运行时出错: {e}"


