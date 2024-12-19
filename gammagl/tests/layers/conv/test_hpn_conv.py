import tensorlayerx as tlx
import numpy as np

from gammagl.layers.conv import HPNConv


def test_hpn_conv():
    x_dict = {
        'author': tlx.random_normal((6, 16), dtype = tlx.float32),
        'paper': tlx.random_normal((5, 12), dtype = tlx.float32)
    }
    edge1 = tlx.convert_to_tensor([[2, 0, 2, 5, 1, 2, 4], [5, 5, 4, 2, 5, 2, 3]])
    edge2 = tlx.convert_to_tensor([[1, 0, 2, 3], [4, 2, 1, 0]])
    edge_index_dict = {
        ('author', 'metapath0', 'author'): edge1,
        ('paper', 'metapath1', 'paper'): edge2,
    }
    num_nodes_dict = {
        'author': 6,
        'paper': 5
    }
    metadata = (list(x_dict.keys()), list(edge_index_dict.keys()))
    in_channels = {'author': 16, 'paper': 12}

    conv = HPNConv(in_channels, 16, metadata, 1)
    out_dict1 = conv(x_dict, edge_index_dict, num_nodes_dict)
    assert len(out_dict1) == 2
    assert tlx.get_tensor_shape(out_dict1['author']) == [6, 16]
    assert tlx.get_tensor_shape(out_dict1['paper']) == [5, 16]