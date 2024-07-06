import tensorflow as tf
import numpy as np
from gammagl.utils.union_utils import convert_union_to_numpy,union_len
def test_union_utils():
    data_tensor = tf.constant([1, 2, 3])
    data_list = [1, 2, 3]
    data_numpy = np.array([1, 2, 3])
    assert np.array_equal(convert_union_to_numpy(data_tensor), np.array([1, 2, 3]))
    assert np.array_equal(convert_union_to_numpy(data_list), np.array([1, 2, 3]))
    assert np.array_equal(convert_union_to_numpy(data_numpy), np.array([1, 2, 3]))
    assert convert_union_to_numpy(data_list, dtype=np.float32).dtype == np.float32
    assert convert_union_to_numpy(None) is None
    assert union_len(data_tensor) == 3
    assert union_len(data_list) == 3
    assert union_len(data_numpy) == 3


