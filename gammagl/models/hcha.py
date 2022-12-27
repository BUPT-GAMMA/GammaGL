import tensorlayerx as tlx
from gammagl.datasets import Planetoid
import numpy as np
from gammagl.layers.conv import MessagePassing
from gammagl.utils.softmax import segment_softmax
import tensorlayerx.nn as nn
from gammagl.layers.conv import HypergraphConv

# from gammagl.layers.conv import HypergraphConv
class HCHA(tlx.nn.Module):
    def __init__(self,
                 in_channels, out_channels, hidden_channels, ea_len, use_attention=False, heads=2,
                 concat=True, negative_slope=0.2, dropout=0.2, bias=True, num_layers = 2, name=None):
        super().__init__(name=name)
        # self.conv1 = HypergraphConv(in_channels, hidden_channels)
        # self.conv2 = HypergraphConv(hidden_channels, out_channels)
        if not use_attention:
            heads = 1
        self.num_layers = num_layers
        if num_layers == 1:
            self.conv = nn.ModuleList([HypergraphConv(in_channels, out_channels, ea_len, 
                                                      heads=heads, use_attention=use_attention)])
        else:
            self.conv = nn.ModuleList([HypergraphConv(in_channels, hidden_channels, ea_len, 
                                                      heads=heads, use_attention=use_attention)])
            for _ in range(1, num_layers - 1):
                self.conv.append(HypergraphConv(heads * hidden_channels, hidden_channels, ea_len, 
                                                      heads=heads, use_attention=use_attention))
            self.conv.append(HypergraphConv(heads * hidden_channels, out_channels, ea_len, 
                                                      use_attention=use_attention))
            # self.conv1 = GCNConv(feature_dim, hidden_dim, norm = 'both')
            # self.conv2 = GCNConv(hidden_dim, num_class, norm = 'both')
            self.relu = tlx.ReLU()
            self.dropout = tlx.layers.Dropout(dropout)


    def forward(self,x, hyperedge_index,hyperedge_weight,hyperedge_attr=None):
        # x = self.conv1(x, hyperedge_index, hyperedge_weight, hyperedge_attr)
        # print('after conv1 out, x.shape:',x.shape)
        # x = self.relu(x)
        # x = self.dropout(x)  # dropout not work in mindspore
        # print('after drop out, x.shape:',x.shape)
        # print('x.shape:',x.shape)
        # print('hyperedge_index:',hyperedge_index.shape)
        # print('hyperedge_weight.shape',hyperedge_weight.shape)
        # print('hyperedge_attr.shape',hyperedge_attr.shape)
        # x = self.conv2(x, hyperedge_index, hyperedge_weight, hyperedge_attr)

        if self.num_layers == 1:
            x = self.conv[0](x, hyperedge_index, hyperedge_weight, hyperedge_attr)
            return x
        for i in range(self.num_layers - 1):
            x = self.conv[i](x, hyperedge_index, hyperedge_weight, hyperedge_attr)
            x = self.relu(x)
            x = self.dropout(x)
        # print('x.shape:',x.shape)
        # print('hyperedge_index:',hyperedge_index.shape)
        # print('hyperedge_weight.shape',hyperedge_weight.shape)
        # print('hyperedge_attr.shape',hyperedge_attr.shape)
        x = self.conv[-1](x, hyperedge_index, hyperedge_weight, hyperedge_attr)
        
        return x

