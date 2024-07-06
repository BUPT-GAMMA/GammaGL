from tensorlayerx import nn
from gammagl.utils import degree
import tensorlayerx as tlx
import numpy as np
from gammagl.layers.attention.edge_encoder import EdgeEncoding
import torch
def test_edge_encoder():
    edge_encoder = EdgeEncoding(edge_dim=3, max_path_distance=4)
    x = torch.randn(5, 10) 
    edge_attr = torch.randn(10, 3)  
    edge_paths = {
        0: {
            1: [0, 1, 2], 
            2: [0, 3]      
        },
        1: {
            2: [1, 2]    
        },
        2: {}           
    }
    cij = edge_encoder(x, edge_attr, edge_paths)
    assert isinstance(cij, torch.Tensor)
    assert cij.shape == (5, 5)
    assert torch.all(torch.isfinite(cij))  

