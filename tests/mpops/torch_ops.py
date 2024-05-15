import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TL_BACKEND'] = 'torch'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
# 0:Output all; 1:Filter out INFO; 2:Filter out INFO and WARNING; 3:Filter out INFO, WARNING, and ERROR

import tensorlayerx as tlx
# import torch
from gammagl.mpops import *
import pytest
import numpy as np

dtype_params = [
    tlx.int8, tlx.int16, tlx.int32, tlx.int64,
    tlx.float16, tlx.float32, tlx.float64
]

def generate_tensor(dim, dtype):
    if dim == 1:
        return tlx.convert_to_tensor([1, 2, 3], dtype=dtype)
    elif dim == 2:
        return tlx.convert_to_tensor([[1, 2], [3, 4], [5, 6]], dtype=dtype)
    elif dim == 3:
        return tlx.convert_to_tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=dtype)


@pytest.mark.parametrize("dtype", dtype_params)
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_segment_max(dtype, dim):
    x = generate_tensor(dim, dtype)
    index = tlx.convert_to_tensor([0, 0, 1], dtype=tlx.int64)
    N = 2
    expected_results = {
        1: tlx.convert_to_tensor([2, 3], dtype=dtype),
        2: tlx.convert_to_tensor([[3, 4], [5, 6]], dtype=dtype),
        3: tlx.convert_to_tensor([[[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=dtype)
    }
    expected_result = expected_results[dim]
    output = segment_max(x, index, N)
    assert np.allclose(tlx.convert_to_numpy(output), tlx.convert_to_numpy(expected_result)), f"Failed with dtype {dtype} and dimension {dim}"


@pytest.mark.parametrize("dtype", dtype_params)
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_segment_sum(dtype, dim):
    x = generate_tensor(dim, dtype)
    index = torch.tensor([0, 0, 1], dtype=torch.int64)
    N = 2
    expected_results = {
        1: tlx.convert_to_tensor([3, 3], dtype=dtype),
        2: tlx.convert_to_tensor([[4, 6], [5, 6]], dtype=dtype),
        3: tlx.convert_to_tensor([[[6, 8], [10, 12]], [[9, 10], [11, 12]]], dtype=dtype)
    }
    expected_result = expected_results[dim]
    output = segment_sum(x, index, N)
    assert np.allclose(tlx.convert_to_numpy(output), tlx.convert_to_numpy(expected_result)), f"Failed with dtype {dtype} and dimension {dim}"


@pytest.mark.parametrize("dtype", dtype_params)
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_segment_mean(dtype, dim):
    x = generate_tensor(dim, dtype)
    index = torch.tensor([0, 0, 1], dtype=torch.int64)
    N = 2
    expected_results = {
        1: tlx.convert_to_tensor([1.5, 3], dtype=dtype),
        2: tlx.convert_to_tensor([[2, 3], [5, 6]], dtype=dtype),
        3: tlx.convert_to_tensor([[[3, 4], [5, 6]], [[9, 10], [11, 12]]], dtype=dtype)
    }
    expected_result = expected_results[dim]
    output = segment_mean(x, index, N)
    assert np.allclose(tlx.convert_to_numpy(output), tlx.convert_to_numpy(expected_result), atol=1e-5), f"Failed with dtype {dtype} and dimension {dim}"
    

# 运行测试
# if __name__ == "__main__":
#     pytest.main([__file__])
