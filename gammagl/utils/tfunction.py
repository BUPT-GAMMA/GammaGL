import numpy as np
from typing import List, Optional, DefaultDict, Union
from collections import defaultdict
import tensorlayerx as tlx
from gammagl.utils.platform_utils import Tensor


class device:
    type: str  # THPDevice_type
    index: int  # THPDevice_index


class dtype:
    # TODO: __reduce__
    is_floating_point: bool
    is_complex: bool
    is_signed: bool
    itemsize: int

def full(size, fill_value, dtype=None):
    if dtype is None:
        numpy_dtype = np.array(fill_value).dtype
    else:
        numpy_dtype = dtype

    result_tensor = np.full(size, fill_value, dtype=numpy_dtype)

    return result_tensor


def zeros(size, dtype=None):
    if dtype is None:
        numpy_dtype = np.array(0).dtype
    else:
        numpy_dtype = dtype
    result_tensor = np.zeros(size, dtype=numpy_dtype)
    return result_tensor


def ones(size, dtype=None):
    if dtype is None:
        numpy_dtype = np.array(1).dtype
    else:
        numpy_dtype = dtype
    result_tensor = np.ones(size, dtype=numpy_dtype)

    return result_tensor


def softmax(input_array, axis=None, dtype=None):
    # 转换为numpy数组
    input_array = np.array(input_array)

    # 为了数值稳定性，减去最大值
    exp_values = np.exp(input_array - np.max(input_array, axis=axis, keepdims=True))

    # 对每个轴上的值进行求和
    softmax_values = exp_values / np.sum(exp_values, axis=axis, keepdims=True)

    # 转换数据类型
    if dtype is not None:
        softmax_values = softmax_values.astype(dtype)

    return softmax_values


def concatenate_tensor(tensors, axis=0):
    # 转换为numpy数组
    tensors = [np.array(tensor) for tensor in tensors]

    # 拼接
    result_tensor = np.concatenate(tensors, axis=axis)

    return result_tensor


def nan_to_num(input_array, nan=0.0, posinf=None, neginf=None):
    # 转换为numpy数组
    input_array = np.array(input_array)

    # 替换
    result_tensor = np.nan_to_num(input_array, nan=nan, posinf=posinf, neginf=neginf)

    return result_tensor


def clip_grad_norm(parameters, max_norm):
    total_norm = 0.0

    # 计算梯度范数
    for param in parameters:
        grad = param.grad.data.numpy()
        total_norm += np.linalg.norm(grad)

    # 计算缩放因子
    clip_coef = max_norm / (total_norm + 1e-6)

    # 缩放每个参数的梯度
    for param in parameters:
        param.grad.data *= clip_coef

    return total_norm


class no_grad:
    def __enter__(self):
        global _requires_grad
        _requires_grad = False

    def __exit__(self, *args):
        global _requires_grad
        _requires_grad = True

def my_function(x):
    if _requires_grad:
        # 在需要计算梯度的情况下，进行梯度计算
        return x * 2
    else:
        # 在不需要计算梯度的情况下，直接返回结果
        return x * 2


class SimpleOptimizer:
    def __init__(self, params, lr=0.01):
        self.params = params
        self.lr = lr
        self.defaults = {'foreach': False, 'fused': False}
        self._zero_grad_profile_name = "zero_grad_profile_name"

    def _patch_step_function(self):
        # 在这里实现 _patch_step_function 的逻辑
        pass

    def _foreach_zero(self, grads):
        # 在这里实现 _foreach_zero 的逻辑
        pass

    def zero_grad(self, set_to_none: bool = True):
        foreach = self.defaults.get('foreach', False) or self.defaults.get('fused', False)

        if not hasattr(self, "_zero_grad_profile_name"):
            self._patch_step_function()

        per_device_and_dtype_grads: Optional[DefaultDict[device, DefaultDict[dtype, List[Tensor]]]]
        if foreach:
            per_device_and_dtype_grads = defaultdict(lambda: defaultdict(list))
        else:
            per_device_and_dtype_grads = None

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        if p.grad.grad_fn is not None:
                            # PyTorch's detach_() is roughly equivalent to NumPy's copy()
                            p.grad = np.copy(p.grad)
                        else:
                            # PyTorch's requires_grad_(False) is equivalent to setting a NumPy array as non-writeable
                            p.grad.setflags(write=0)

                        if (not foreach or p.grad.is_sparse):
                            p.grad.fill(0)
                        else:
                            assert per_device_and_dtype_grads is not None
                            per_device_and_dtype_grads[p.grad.device][np.dtype(p.grad.dtype)].append(p.grad)

        if foreach:
            assert per_device_and_dtype_grads is not None
            for per_dtype_grads in per_device_and_dtype_grads.values():
                for grads in per_dtype_grads.values():
                    self._foreach_zero(grads)


if __name__ == '__main__':
    print(full((3, 4), 1))
    print(zeros(3, 4))
    print(ones(3, 4))