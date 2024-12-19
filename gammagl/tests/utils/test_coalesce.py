import tensorlayerx as tlx
from gammagl.utils import coalesce


def test_coalesce():
    edge_index = tlx.convert_to_tensor([[0, 1, 0, 1],
                                       [1, 0, 1, 0]])
    attr = tlx.convert_to_tensor([[1, 0],
                                 [2, 3],
                                 [-5, 4],
                                 [0, 2]], dtype=tlx.float32)

    edge_index = coalesce(edge_index)
    assert tlx.ops.convert_to_numpy(edge_index).tolist() == [[0, 1], [1, 0]]

    edge_index = tlx.convert_to_tensor([[0, 1, 0, 1],
                                       [1, 0, 1, 0]])
    edge_index, attr = coalesce(edge_index, attr)
    assert tlx.ops.convert_to_numpy(edge_index).tolist() == [[0, 1], [1, 0]]
    assert tlx.ops.convert_to_numpy(attr).tolist() == [[-4.0, 4.0], [2.0, 5.0]]
