# -*- encoding: utf-8 -*-
"""
@File   : jknet.py
@Time   : 2022/4/10 11:16 上午
@Author : Jia Yiming
"""

import tensorlayerx as tlx
from gammagl.layers.conv import SAGEConv, GCNConv, JumpingKnowledge


class JKNet(tlx.nn.Module):
    def __init__(self, dataset, mode='max', num_layers=6, hidden=16, drop=0.5):
        super(JKNet, self).__init__()
        self.num_layers = num_layers
        self.mode = mode
        self.drop = drop

        self.conv0 = GCNConv(in_channels=dataset.num_node_features, out_channels=hidden)
        self.dropout0 = tlx.nn.Dropout(p=drop)
        for i in range(1, self.num_layers):
            setattr(self, 'conv{}'.format(i), GCNConv(hidden, hidden))
            setattr(self, 'dropout{}'.format(i), tlx.nn.Dropout(p=drop))
        self.jk = JumpingKnowledge(mode=mode, channels=hidden, num_layers=num_layers)
        if mode == 'max':
            self.fc = tlx.nn.Linear(in_features=hidden, out_features=dataset.num_classes)
        elif mode == 'cat':
            self.fc = tlx.nn.Linear(in_features=num_layers * hidden, out_features=dataset.num_classes)
        elif mode == 'lstm':
            self.fc = tlx.nn.Linear(in_features=hidden, out_features=dataset.num_classes)


    def forward(self, x, edge_index, edge_weight, num_nodes):
        layer_out = []
        for i in range(self.num_layers):
            conv = getattr(self, 'conv{}'.format(i))
            dropout = getattr(self, 'dropout{}'.format(i))
            x = dropout(tlx.relu(conv(x, edge_index, edge_weight=edge_weight)))
            layer_out.append(x)
        h = self.jk(layer_out)
        h = self.fc(h)
        h = tlx.softmax(h, axis=1)
        return h