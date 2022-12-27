import tensorlayerx as tlx

from gammagl.layers.conv import HypergraphConv


def test_hypergraph_conv_with_more_nodes_than_edges():
    in_channels, out_channels = (16, 32)
    hyperedge_index = tlx.convert_to_tensor([[0, 0, 1, 1, 2, 3], [0, 1, 0, 1, 0, 1]])
    hyperedge_weight = tlx.ones((len(hyperedge_index[1]),))
    num_nodes = max(hyperedge_index[0]) + 1
    x = tlx.random_normal(shape=(num_nodes, in_channels))
    hyperedge_attr = tlx.random_normal(shape=(max(hyperedge_index[0]) + 1, in_channels))
    ea_len = len(hyperedge_attr[0])

    conv = HypergraphConv(in_channels, out_channels, ea_len)
    out = conv(x, hyperedge_index, hyperedge_weight, hyperedge_attr)
    assert tlx.get_tensor_shape(out) == [num_nodes, out_channels]

    conv = HypergraphConv(in_channels, out_channels, ea_len, use_attention=True,
                          heads=2)
    out = conv(x, hyperedge_index, hyperedge_weight, hyperedge_attr)
    assert tlx.get_tensor_shape(out) == [num_nodes, 2 * out_channels]

    conv = HypergraphConv(in_channels, out_channels, ea_len, use_attention=True,
                          heads=2, concat=False, dropout=0.5)
    out = conv(x, hyperedge_index, hyperedge_weight, hyperedge_attr)
    assert tlx.get_tensor_shape(out) == [num_nodes, out_channels]


def test_hypergraph_conv_with_more_edges_than_nodes():
    in_channels, out_channels = (16, 32)
    hyperedge_index = tlx.convert_to_tensor([[0, 0, 1, 1, 2, 3, 3, 3, 2, 1, 2],
                                             [0, 1, 2, 1, 2, 1, 0, 3, 3, 4, 5]])
    hyperedge_weight = tlx.ones((len(hyperedge_index[1]),))
    num_nodes = max(hyperedge_index[0]) + 1
    x = tlx.random_normal(shape=(num_nodes, in_channels))
    hyperedge_attr = tlx.random_normal(shape=(max(hyperedge_index[1]) + 1, in_channels))
    ea_len = len(hyperedge_attr[0])

    conv = HypergraphConv(in_channels, out_channels, ea_len)
    out = conv(x, hyperedge_index, hyperedge_weight, hyperedge_attr)
    assert tlx.get_tensor_shape(out) == [num_nodes, out_channels]