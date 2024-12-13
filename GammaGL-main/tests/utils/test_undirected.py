import tensorlayerx as tlx
from gammagl.utils import to_undirected, is_undirected


def test_is_undirected():
    row = tlx.convert_to_tensor([0, 1, 0])
    col = tlx.convert_to_tensor([1, 0, 0])
    sym_weight = tlx.convert_to_tensor([0, 0, 1])
    asym_weight = tlx.convert_to_tensor([0, 1, 1])

    assert is_undirected(tlx.stack([row, col], axis=0))
    assert is_undirected(tlx.stack([row, col], axis=0), sym_weight)
    assert not is_undirected(tlx.stack([row, col], axis=0), asym_weight)

    row = tlx.convert_to_tensor([0, 1, 1])
    col = tlx.convert_to_tensor([1, 0, 2])

    assert not is_undirected(tlx.stack([row, col], axis=0))


def test_to_undirected():
    row = tlx.convert_to_tensor([0, 1, 1])
    col = tlx.convert_to_tensor([1, 0, 2])
    attr = tlx.convert_to_tensor([[1, 0],
                                 [2, 3],
                                 [-5, 4]], dtype=tlx.float32)

    edge_index, attr = to_undirected(edge_index=tlx.stack([row, col], axis=0), edge_attr=attr, reduce='add')
    assert tlx.ops.convert_to_numpy(edge_index).tolist() == [[0, 1, 1, 2], [1, 0, 2, 1]]
    assert tlx.ops.convert_to_numpy(attr).tolist() == [[3.,  3.],
                                                       [3.,  3.],
                                                       [-5.,  4.],
                                                       [-5.,  4.]]
