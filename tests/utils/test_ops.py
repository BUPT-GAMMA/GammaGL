import numpy as np
import paddle
from gammagl.utils.ops import get_index_from_counts

def test_get_index_from_counts():
    # 构造测试用例
    counts_np = np.array([2, 3, 4])  # numpy 数组作为输入
    counts_pd = paddle.to_tensor(counts_np)  # 转换为 paddle.Tensor

    # 期望的输出
    expected_index_np = np.array([0, 2, 5, 9])
    expected_index_pd = paddle.to_tensor(expected_index_np)

    # 调用函数计算结果
    index_np = get_index_from_counts(counts_np)
    index_pd = get_index_from_counts(counts_pd)

    # 使用 assert 语句验证结果
    assert np.array_equal(index_np, expected_index_np)
    assert paddle.allclose(index_pd, expected_index_pd)


