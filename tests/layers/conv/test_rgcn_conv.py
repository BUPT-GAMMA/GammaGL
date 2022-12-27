from itertools import product

import pytest
import tensorlayerx as tlx
import numpy as np

from gammagl.layers.conv import RGCNConv

classes = [RGCNConv]
confs = [(None, None), (2, None), (None, 2)]

@pytest.mark.parametrize('cls,conf', product(classes, confs))
def test_rgcn_conv(cls, conf):
    num_bases, num_blocks = conf

    x1 = tlx.random_normal((4, 4))
    x2 = tlx.random_normal((2, 16))
    idx1 = tlx.arange(start = 0, limit = 4, dtype = tlx.int64)
    idx2 = tlx.arange(start = 0, limit = 2, dtype = tlx.int64)
    edge_index = tlx.convert_to_tensor([[0, 1, 1, 2, 2, 3], [0, 0, 1, 0, 1, 1]], dtype = tlx.int64)
    edge_type = tlx.convert_to_tensor([0, 1, 1, 0, 0, 1], dtype = tlx.int64)
    row, col = edge_index

    conv = cls(4, 32, 2, num_bases = num_bases, num_blocks = num_blocks)
    out1 = conv(x1, edge_index, edge_type)
    assert tlx.get_tensor_shape(out1) == [4, 32]

    if num_blocks is None:
        out2 = conv(None, edge_index, edge_type)
        assert tlx.get_tensor_shape(out2) == [4, 32]

    conv = cls((4, 16), 32, 2, num_bases = num_bases, num_blocks = num_blocks)
    out1 = conv((x1, x2), edge_index, edge_type)
    assert tlx.get_tensor_shape(out1) == [2, 32]

    if num_blocks is None:
        out2 = conv((None, idx2), edge_index, edge_type)
        assert tlx.get_tensor_shape(out2) == [2, 32]
        assert np.allclose(tlx.convert_to_numpy(conv((idx1, idx2), edge_index, edge_type)), 
                           tlx.convert_to_numpy(out2))
