import time
import resource
import numpy as np
import os
os.environ['TL_BACKEND'] = 'torch'

# ===================== 核心替换：PyTorch → TensorLayerX =====================
import tensorlayerx as tlx
from tensorlayerx.nn import Module


class F1Calculator(object):
    """F1分数计算：纯 TLX 实现，跨后端完全兼容"""
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        # 【修复1】不要提前使用 tlx.zeros 初始化（避免默认分配到 CPU 导致设备冲突）
        # 初始化为 0.0，在第一次 update 时会自动转变为对应设备上的 Tensor
        self.TP = 0.0
        self.FP = 0.0
        self.FN = 0.0

    def update(self, y_true, y_pred):
        # 【修复2】跨后端极其稳定的纯 TLX one-hot 实现（避开版本迭代导致的 API 缺失）
        def _to_one_hot(tensor, num_classes):
            if len(tensor.shape) == 1 or tensor.shape[1] == 1:
                # 展平并转为整数索引
                idx = tlx.cast(tlx.reshape(tensor, (-1,)), tlx.int64)
                
                # 使用 np.eye 构建单位阵并转为 Tensor
                eye_np = np.eye(num_classes, dtype=np.float32)
                eye_tlx = tlx.convert_to_tensor(eye_np)
                
                # 动态放置到当前数据的设备上（防 CPU/GPU 冲突）
                if hasattr(tensor, 'device'):
                    eye_tlx = eye_tlx.to(tensor.device)
                    
                # 通过 gather 索引实现 one-hot，非常高效且完全跨后端
                return tlx.gather(eye_tlx, idx)
            return tlx.cast(tensor, tlx.float32)

        # 转换数据
        y_true = _to_one_hot(y_true, self.num_classes)
        y_pred = _to_one_hot(y_pred, self.num_classes)
        
        # 统计结果，使用纯 TLX 函数
        self.TP += tlx.reduce_sum(y_true * y_pred, axis=0)
        self.FP += tlx.reduce_sum((1 - y_true) * y_pred, axis=0)
        self.FN += tlx.reduce_sum(y_true * (1 - y_pred), axis=0)

    def compute(self, average: str=None):
        eps = 1e-10
        
        # 如果从没有进入过 update，避免计算错误
        if isinstance(self.TP, float):
            return 0.0
            
        TP = self.TP
        FP = self.FP
        FN = self.FN
        
        if average == 'micro':
            f1 = 2 * tlx.reduce_sum(TP) / (2 * tlx.reduce_sum(TP) + tlx.reduce_sum(FP) + tlx.reduce_sum(FN) + eps)
            return float(f1)
        elif average == 'macro':
            f1 = 2 * TP / (2 * TP + FP + FN + eps)
            return float(tlx.reduce_mean(f1))
        else:
            raise ValueError('average must be "micro" or "macro"')


class Stopwatch(object):
    """【纯Python计时工具，无修改】"""
    def __init__(self):
        self.reset()

    def start(self):
        self.start_time = time.time()

    def pause(self) -> float:
        self.elapsed_sec += time.time() - self.start_time
        self.start_time = None
        return self.elapsed_sec

    def lap(self) -> float:
        return time.time() - self.start_time + self.elapsed_sec

    def reset(self):
        self.start_time = None
        self.elapsed_sec = 0

    @property
    def time(self) -> float:
        return self.elapsed_sec


class Accumulator(object):
    """【纯Python累加工具，无修改】"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.count = 0

    def update(self, val: float, count: int=1):
        self.val += val
        self.count += count
        return self.val

    @property
    def avg(self) -> float:
        return self.val / self.count


def get_ram() -> float:
    """【纯Python内存获取，无修改】"""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def get_cuda_mem(dev) -> float:
    """TLX 兼容 CUDA 内存统计"""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated(dev) / 2**30
    except ImportError:
        pass
    return 0.0


def get_num_params(model: Module) -> float:
    """参数统计：【修复3】改为纯 TLX 标准接口，弃用 torch 的 model.parameters()"""
    # TLX 中获取网络所有可训练权重是 model.trainable_weights
    num_paramst = sum([np.prod(param.shape) for param in model.trainable_weights])
    return num_paramst / 1e6


def get_mem_params(model: Module) -> float:
    """模型内存统计：【修复4】改为纯 TLX 标准接口"""
    # 按照标准 float32 为 4 字节计算大小
    mem_params = sum([np.prod(param.shape) * 4 for param in model.all_weights])
    return mem_params / (1024**2)