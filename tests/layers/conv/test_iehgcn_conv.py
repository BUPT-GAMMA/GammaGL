import tensorlayerx as tlx
from gammagl.layers.conv import MessagePassing, GCNConv
from tensorlayerx.nn import ModuleDict, Linear, Dropout
from tensorlayerx import elu
from gammagl.layers.conv import ieHGCNConv
import tensorlayerx as tlx
import numpy as np

def test_iehgcn_conv():
    num_nodes_dict = {
        'user': 5,
        'item': 4
    }
    in_channels_dict = {
        'user': 6,
        'item': 6
    }
    out_channels = 8
    attn_channels = 4
    metadata = (['user', 'item'], [('user', 'to', 'item'), ('item', 'to', 'user')])
    x_dict = {
        'user': tlx.convert_to_tensor(np.random.randn(num_nodes_dict['user'], in_channels_dict['user']), dtype=tlx.float32),
        'item': tlx.convert_to_tensor(np.random.randn(num_nodes_dict['item'], in_channels_dict['item']), dtype=tlx.float32)
    }
    edge_index_dict = {
        ('user', 'to', 'item'): tlx.convert_to_tensor(np.array([
            [0, 1, 2, 3],
            [0, 1, 2, 3]
        ]), dtype=tlx.int64),
        ('item', 'to', 'user'): tlx.convert_to_tensor(np.array([
            [0, 1, 2, 3],
            [0, 1, 2, 3]
        ]), dtype=tlx.int64)
    }
    conv = ieHGCNConv(in_channels=in_channels_dict, out_channels=out_channels, attn_channels=attn_channels, metadata=metadata)
    try:
        out_dict = conv(x_dict, edge_index_dict, num_nodes_dict)
    except Exception as e:
        assert False, f"运行时出错: {e}"

