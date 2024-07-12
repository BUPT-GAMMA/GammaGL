import tensorlayerx as tlx
from gammagl.layers.conv.compgcn_conv import  CompConv


def test_compgcn_conv():
    conv = CompConv(in_channels=64, out_channels=128, num_relations=3)
    x = tlx.random_normal(shape=(3, 64)) 
    edge_index = tlx.convert_to_tensor([[0, 1, 1, 2], [1, 0, 2, 1]])  
    edge_type = tlx.convert_to_tensor([0, 1, 2, 3])  
    ref_emb = tlx.random_normal(shape=(4, 64))
    output, ref_emb = conv(x, edge_index, edge_type, ref_emb)
    assert output.shape == (3, 128)  
