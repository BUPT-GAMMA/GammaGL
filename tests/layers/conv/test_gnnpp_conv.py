#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import tensorlayerx as tlx
from gammagl.data import Graph
from gammagl.layers.conv import GNNPPConv

def test_gnnpp_conv():
    num_nodes = 4
    num_edges = 6
    hidden_dim = 16
    x = tlx.random_normal(shape=(num_nodes, hidden_dim))
    edge_attr = tlx.random_normal(shape=(num_edges, hidden_dim))
    edge_index = tlx.convert_to_tensor(
        [[0, 1, 2, 3, 0, 2],
         [1, 2, 3, 0, 2, 1]],
        dtype=tlx.int64
    )
    batch = Graph(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr
    )
    conv = GNNPPConv(
        in_dim=hidden_dim,
        out_dim=hidden_dim,
        dropout=0.0,
        residual=True,
        ffn=True
    )
    out = conv(batch)
    assert out.x.shape == (num_nodes, hidden_dim)
    assert out.edge_attr.shape == (num_edges, hidden_dim)
