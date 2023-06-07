import tensorlayerx as tlx
import numpy as np
import pytest

from gammagl.layers.conv import HANConv


def test_han_conv():
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

    conv = HANConv(in_channels, 16, metadata, heads=2)
    out_dict1 = conv(x_dict, edge_index_dict, num_nodes_dict)
    assert len(out_dict1) == 2
    assert tlx.get_tensor_shape(out_dict1['author']) == [6, 32]
    assert tlx.get_tensor_shape(out_dict1['paper']) == [5, 32]

    # non zero dropout
    conv = HANConv(in_channels, 16, metadata, heads=2, dropout_rate=0.1)
    out_dict1 = conv(x_dict, edge_index_dict, num_nodes_dict)
    assert len(out_dict1) == 2
    assert tlx.get_tensor_shape(out_dict1['author']) == [6, 32]
    assert tlx.get_tensor_shape(out_dict1['paper']) == [5, 32]

def test_han_conv_empty_tensor():
    x_dict = {
        'author': tlx.random_normal((6, 16), dtype = tlx.float32),
        'paper': tlx.random_normal((0, 12), dtype = tlx.float32),
    }
    edge1 = np.random.randint(0, 0, size = (2, 0), dtype = np.int64)
    edge2 = np.random.randint(0, 0, size = (2, 0), dtype = np.int64)
    edge3 = np.random.randint(0, 0, size = (2, 0), dtype = np.int64)
    edge_index_dict = {
        ('paper', 'to', 'author'): tlx.convert_to_tensor(edge1),
        ('author', 'to', 'paper'): tlx.convert_to_tensor(edge2),
        ('paper', 'to', 'paper'): tlx.convert_to_tensor(edge3),
    }
    num_nodes_dict = {
        'author': 6,
        'paper': 0
    }
    metadata = (list(x_dict.keys()), list(edge_index_dict.keys()))
    in_channels = {'author': 16, 'paper': 12}
    conv = HANConv(in_channels, 16, metadata, heads=2)
    if tlx.BACKEND == 'paddle':
        # [Note]: paddle backend do not support empty feature input
        return
    else:
        out_dict = conv(x_dict, edge_index_dict, num_nodes_dict)
        assert len(out_dict) == 2
        assert tlx.get_tensor_shape(out_dict['author'])== [6, 32]
        # [TODO]: check the feature of the 'author'
        # if tlx.BACKEND in ('torch', 'mindspore'):
        #     assert tlx.all(out_dict['author'] == 0)
        # assert pytest.approx(tlx.convert_to_numpy(out_dict['author'])) == 0.0
        assert tlx.get_tensor_shape(out_dict['paper']) == [0, 32]
