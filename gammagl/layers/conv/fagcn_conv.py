# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File   :  fagcn_conv.py
@Time   :  2022/5/10 11:02
@Author :  Ma Zeyao
"""

import tensorlayerx as tlx
import tensorlayerx.nn as nn
from gammagl.layers.conv import MessagePassing


class FAGCNConv(MessagePassing):
    r"""The Frequency Adaptive Graph Convolution operator from the
    `"Beyond Low-Frequency Information in Graph Convolutional Networks"
    <https://arxiv.org/abs/2101.00797>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i= \epsilon \cdot \mathbf{x}^{(0)}_i +
        \sum_{j \in \mathcal{N}(i)} \frac{\alpha_{i,j}}{\sqrt{d_i d_j}}
        \mathbf{x}_{j}

    where :math:`\mathbf{x}^{(0)}_i` and :math:`d_i` denote the initial feature
    representation and node degree of node :math:`i`, respectively.
    The attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \mathbf{\alpha}_{i,j} = \textrm{tanh}(\mathbf{a}^{\top}[\mathbf{x}_i,
        \mathbf{x}_j])

    based on the trainable parameter vector :math:`\mathbf{a}`.

    Parameters
    ----------
    hidden_dim: int
        Hidden dimension of layer
    drop_rate: float
        Dropout rate
    """

    def __init__(self, hidden_dim, drop_rate):
        super(FAGCNConv, self).__init__()
        self.dropout = nn.Dropout(drop_rate)
        self.tanh = nn.Tanh()

        init_w = nn.initializers.XavierNormal(gain=1.414)
        init_b = nn.initializers.random_uniform(-hidden_dim**-0.5, hidden_dim**-0.5)
        self.att_src = self._get_weights("att_src", shape=(1, hidden_dim), init=init_w)
        self.att_dst = self._get_weights("att_dst", shape=(1, hidden_dim), init=init_w)
        self.bias_src = self._get_weights("bias_src", shape=(hidden_dim, ), init=init_b)
        self.bias_dst = self._get_weights("bias_dst", shape=(hidden_dim, ), init=init_b)

    def message(self, x, edge_index, edge_weight=None, num_nodes=None):
        node_src = edge_index[0, :]
        node_dst = edge_index[1, :]

        weight_src = tlx.ops.gather(tlx.ops.reduce_sum(x * self.att_src + self.bias_src, -1), node_src)
        weight_dst = tlx.ops.gather(tlx.ops.reduce_sum(x * self.att_dst + self.bias_dst, -1), node_dst)
        weight = self.tanh(weight_src + weight_dst)

        weight = weight * edge_weight
        alpha = self.dropout(weight)

        x = tlx.ops.gather(x, node_src) * tlx.ops.expand_dims(alpha, -1)
        return x

    def forward(self, x, edge_index, edge_weight, num_nodes):
        x = self.propagate(x, edge_index, edge_weight=edge_weight, num_nodes=num_nodes)
        return x
