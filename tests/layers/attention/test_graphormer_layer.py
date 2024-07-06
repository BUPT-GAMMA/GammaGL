from tensorlayerx import nn
from gammagl.utils import degree
import tensorlayerx as tlx
import numpy as np
from gammagl.layers.attention.edge_encoder import EdgeEncoding
from gammagl.layers.attention.graphormer_layer import GraphormerLayer
import torch
def test_graphormer_layer():
    layer = GraphormerLayer(node_dim=64, edge_dim=16, n_heads=8, max_path_distance=5)
    x = torch.randn(32, 64)  
    edge_attr = torch.randn(64, 16) 
    b = torch.randn(32, 32) 
    edge_paths = {
        0: {1: [0, 1, 2]},  
        1: {2: [1, 2]},    
        2: {}              
    }
    ptr = None  
    output = layer(x, edge_attr, b, edge_paths, ptr)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (32, 64) 

    
