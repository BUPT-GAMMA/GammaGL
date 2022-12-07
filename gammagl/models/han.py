# !/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2022/04/21 20:30
# @Author  : clear
# @FileName: han.py

import tensorlayerx as tlx
from gammagl.layers.conv import HANConv

class HAN(tlx.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 metadata,
                 drop_rate,
                 hidden_channels=128,
                 heads=8,
                 name=None):
        super().__init__(name=name)
        self.han_conv = HANConv(in_channels,
                                hidden_channels,
                                metadata,
                                heads=heads,
                                dropout_rate=drop_rate)
        self.lin = tlx.nn.Linear(in_features=hidden_channels*heads,
                                 out_features=out_channels)


    def forward(self, x_dict, edge_index_dict, num_nodes_dict):
        x = self.han_conv(x_dict, edge_index_dict, num_nodes_dict)
        out = {}
        for node_type, _ in num_nodes_dict.items():
            out[node_type] = self.lin(x[node_type])
        return out
