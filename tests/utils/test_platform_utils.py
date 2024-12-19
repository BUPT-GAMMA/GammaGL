# -*- coding: utf-8 -*-
# @author WuJing
# @created 2023/4/19


from gammagl.utils.platform_utils import *


def test_with_dtype():
    arr = np.array([1, 2, 3], dtype=np.int32)
    assert with_dtype(arr, np.int64).dtype == np.int64
    t = tlx.convert_to_tensor([1, 2, 3], dtype=tlx.int32)
    assert with_dtype(t, tlx.int64).dtype == tlx.int64

    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert with_dtype(arr, np.float64).dtype == np.float64
    t = tlx.convert_to_tensor([1.0, 2.0, 3.0], dtype=tlx.float32)
    assert with_dtype(t, tlx.float64).dtype == tlx.float64

