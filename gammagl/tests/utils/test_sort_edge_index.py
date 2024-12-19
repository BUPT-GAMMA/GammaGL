import tensorlayerx as tlx

from gammagl.utils import sort_edge_index


def test_sort_edge_index():
    edge_index = tlx.convert_to_tensor([[2, 1, 1, 0], [1, 2, 0, 1]])
    edge_attr = tlx.convert_to_tensor([[1], [2], [3], [4]])

    out = sort_edge_index(edge_index)
    assert tlx.ops.convert_to_numpy(out).tolist() == [[0, 1, 1, 2], [1, 0, 2, 1]]

    out = sort_edge_index(edge_index, edge_attr)
    assert tlx.ops.convert_to_numpy(out[0]).tolist() == [[0, 1, 1, 2], [1, 0, 2, 1]]
    assert tlx.ops.convert_to_numpy(out[1]).tolist() == [[4], [3], [2], [1]]
