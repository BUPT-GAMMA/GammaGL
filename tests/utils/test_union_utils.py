import tensorflow as tf
import numpy as np
from gammagl.utils.union_utils import convert_union_to_numpy,union_len
def test_union_utils():
    # 测试 convert_union_to_numpy 函数
    data_tensor = tf.constant([1, 2, 3])
    data_list = [1, 2, 3]
    data_numpy = np.array([1, 2, 3])

    # 张量转换为 NumPy 数组
    assert np.array_equal(convert_union_to_numpy(data_tensor), np.array([1, 2, 3]))

    # 列表转换为 NumPy 数组
    assert np.array_equal(convert_union_to_numpy(data_list), np.array([1, 2, 3]))

    # NumPy 数组不变
    assert np.array_equal(convert_union_to_numpy(data_numpy), np.array([1, 2, 3]))

    # 数据类型转换
    assert convert_union_to_numpy(data_list, dtype=np.float32).dtype == np.float32

    # None 检查
    assert convert_union_to_numpy(None) is None

    # 测试 union_len 函数
    # 张量长度
    assert union_len(data_tensor) == 3

    # 列表长度
    assert union_len(data_list) == 3

    # NumPy 数组长度
    assert union_len(data_numpy) == 3


