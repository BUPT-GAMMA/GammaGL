# -*- coding: utf-8 -*-
# @author WuJing
# @created 2022/11/11

import tensorlayerx as tlx
from gammagl.layers.conv import FILMConv


def test_film_conv():
    x1 = tlx.random_normal((4, 4))
    # x2 = tlx.random_normal((2, 16))
    edge_index = tlx.convert_to_tensor([[0, 1, 1, 2, 2, 3], [0, 0, 1, 0, 1, 1]])

    conv = FILMConv(4, 32)
    out1 = conv(x1, edge_index)

    assert tlx.get_tensor_shape(out1) == [4, 32]
