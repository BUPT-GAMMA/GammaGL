import numpy as np
import paddle
from gammagl.utils.ops import get_index_from_counts

def test_get_index_from_counts():
    counts_np = np.array([2, 3, 4]) 
    counts_pd = paddle.to_tensor(counts_np) 
    expected_index_np = np.array([0, 2, 5, 9])
    expected_index_pd = paddle.to_tensor(expected_index_np)
    index_np = get_index_from_counts(counts_np)
    index_pd = get_index_from_counts(counts_pd)
    assert np.array_equal(index_np, expected_index_np)
    assert paddle.allclose(index_pd, expected_index_pd)


