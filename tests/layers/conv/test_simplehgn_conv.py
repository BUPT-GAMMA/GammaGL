import tensorlayerx as tlx
from gammagl.layers.conv import SimpleHGNConv


def test_simplehgn_conv():
    x = tlx.random_normal(shape=(4, 64))
    edge_index = tlx.convert_to_tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    edge_feat = tlx.convert_to_tensor([0, 1, 2, 3, 4, 5])
    
    conv = SimpleHGNConv(in_feats=64, out_feats=128, num_etypes=6, edge_feats=64, heads=8)
    out, _ = conv(x, edge_index, edge_feat=edge_feat)
    assert tlx.get_tensor_shape(out) == [4, 8, 128]
