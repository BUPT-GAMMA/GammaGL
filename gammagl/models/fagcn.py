# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File   :  fagcn.py
@Time   :  2022/5/10 11:02
@Author :  Ma Zeyao
"""

import tensorlayerx.nn as nn
from gammagl.layers.conv.fagcn_conv import FAGCNConv


class FAGCNModel(nn.Module):
    r"""The Frequency Adaptive Graph Convolution operator from the
    `"Beyond Low-Frequency Information in Graph Convolutional Networks"
    <https://arxiv.org/abs/2101.00797>`_ paper

    Parameters
    ----------
    feature_dim: int
        Dimension of feature vector in original input.
    hidden_dim: int
        Dimension of feature vector in FAGCN.
    num_class: int
        Dimension of feature vector in forward output.
    drop_rate: float
        Dropout rate.
    eps: float
        Epsilon parameter in paper
    num_layers: int
        Number of Frequency Adaptation Graph Convolutional Layers
    """

    def __init__(self, feature_dim, hidden_dim,
                 num_class, drop_rate, eps, num_layers, name=None):
        super().__init__(name=name)
        self.drop_rate = drop_rate
        self.eps = eps
        self.num_layers = num_layers
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drop_rate)

        self.mid_layers = nn.Sequential()
        for i in range(num_layers):
            self.mid_layers.append(FAGCNConv(hidden_dim=hidden_dim, drop_rate=drop_rate))

        init = nn.initializers.XavierNormal(gain=1.414)
        self.first_layer = nn.Linear(in_features=feature_dim, out_features=hidden_dim, W_init=init)
        self.last_layer = nn.Linear(in_features=hidden_dim, out_features=num_class, W_init=init)


    def forward(self, x, edge_index, edge_weight, num_nodes):
        x = self.dropout(x)
        x = self.relu(self.first_layer(x))
        x = self.dropout(x)
        x_0 = x
        for i in range(self.num_layers):
            x = self.mid_layers[i](x, edge_index, edge_weight, num_nodes)
            x = self.eps * x_0 + x
        x = self.last_layer(x)

        return x
