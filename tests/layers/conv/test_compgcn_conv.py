import tensorlayerx as tlx
import numpy as np
import torch
from gammagl.layers.conv import MessagePassing

from gammagl.mpops import *
from gammagl.layers.conv.compgcn_conv import  CompConv
def test_compgcn_conv():
    conv = CompConv(in_channels=64, out_channels=128, num_relations=4)
    x = torch.randn(32, 64) 
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])  
    edge_type = torch.tensor([0, 1, 2, 3])  
    ref_emb = torch.randn(4, 64)
    output, ref_emb = conv(x, edge_index, edge_type, ref_emb)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (32, 128)  
