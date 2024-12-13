import tensorlayerx as tlx
import pytest
from gammagl.utils import to_dense_batch

def test_to_dense_batch():
    x = tlx.convert_to_tensor([[1,2], [3,4], [5,6], [6,7]])
    batch = tlx.convert_to_tensor([0, 1, 1, 1])
    a, b = to_dense_batch(x, batch)
    assert tlx.convert_to_numpy(a).tolist() == [[[1,2],[0,0],[0,0]],[[3,4],[5,6],[6,7]]]
    assert tlx.convert_to_numpy(b).tolist() == [[True, False, False], [True, True, True]]

    x = tlx.convert_to_tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
    batch = tlx.convert_to_tensor([0, 0, 1, 2, 2, 2])

    expected = tlx.convert_to_tensor([
        [[1, 2], [3, 4], [0, 0]],
        [[5, 6], [0, 0], [0, 0]],
        [[7, 8], [9, 10], [11, 12]],
    ])

    out, mask = to_dense_batch(x, batch)
    assert tlx.get_tensor_shape(out) == [3, 3, 2]
    assert pytest.approx(out) == expected
    # assert torch.equal(out, expected)
    assert tlx.convert_to_numpy(mask).tolist() == [[1, 1, 0], [1, 0, 0], [1, 1, 1]]

    # out, mask = to_dense_batch(x, batch, max_num_nodes=2)
    # assert tlx.get_tensor_shape(out) == (3, 2, 2)
    # assert pytest.approx(out) == expected[:, :2]
    # assert tlx.convert_to_numpy(mask).tolist() == [[1, 1], [1, 0], [1, 1]]

    # out, mask = to_dense_batch(x, batch, max_num_nodes=5)
    # assert out.size() == (3, 5, 2)
    # assert torch.equal(out[:, :3], expected)
    # assert mask.tolist() == [[1, 1, 0, 0, 0], [1, 0, 0, 0, 0], [1, 1, 1, 0, 0]]

    # out, mask = to_dense_batch(x)
    # assert out.size() == (1, 6, 2)
    # assert torch.equal(out[0], x)
    # assert mask.tolist() == [[1, 1, 1, 1, 1, 1]]

    # out, mask = to_dense_batch(x, max_num_nodes=2)
    # assert out.size() == (1, 2, 2)
    # assert torch.equal(out[0], x[:2])
    # assert mask.tolist() == [[1, 1]]

    # out, mask = to_dense_batch(x, max_num_nodes=10)
    # assert out.size() == (1, 10, 2)
    # assert torch.equal(out[0, :6], x)
    # assert mask.tolist() == [[1, 1, 1, 1, 1, 1, 0, 0, 0, 0]]

    # out, mask = to_dense_batch(x, batch, batch_size=4)
    # assert out.size() == (4, 3, 2)
    