# !/usr/bin/env python3
# -*- coding:utf-8 -*-

import tensorlayerx as tlx
from gammagl.layers.conv import HPNConv

class HPN(tlx.nn.Module):
    r'''HPN proposed in `"Heterogeneous Graph Propagation Network"
        <https://ieeexplore.ieee.org/abstract/document/9428609>`_ paper.


        Parameters
        ----------
        in_channels: int
            input dimension of the feature.
        out_channels: int
            output dimension of the feature.
        metadata: Tuple[List[str], List[Tuple[str, str, str]]]
            the metadata of the heterogeneous graph.
        drop_rate: float
            dropout probability.
        iter_K: int
            number of iteration used in APPNPConv.
        hidden_channels: int, optional
            hidden dimension of the feature.
        alpha: float, optional
            parameters used in APPNPConv.
        name: str, optional
            model name.

    '''
    def __init__(self,
                 in_channels,
                 out_channels,
                 metadata,
                 drop_rate,
                 iter_K,
                 hidden_channels=128,
                 alpha=0.2,
                 name=None):
                         
                 
        super().__init__(name=name)
        self.hpn_conv = HPNConv(in_channels,
                                hidden_channels,
                                metadata,
                                iter_K=iter_K,
                                alpha=alpha,
                                drop_rate=drop_rate)
        self.lin = tlx.nn.Linear(in_features=hidden_channels,
                                 out_features=out_channels)


    def forward(self, x_dict, edge_index_dict, num_nodes_dict):
        x = self.hpn_conv(x_dict, edge_index_dict, num_nodes_dict)
        out = {}
        for node_type, _ in num_nodes_dict.items():
            out[node_type] = self.lin(x[node_type])
        return out
