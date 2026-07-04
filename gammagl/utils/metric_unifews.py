import time
import resource
import numpy as np
import os
import tensorlayerx as tlx
from tensorlayerx.nn import Module


class F1Calculator(object):
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.TP = 0.0
        self.FP = 0.0
        self.FN = 0.0

    def update(self, y_true, y_pred):
        def _to_one_hot(tensor, num_classes):
            if len(tensor.shape) == 1 or tensor.shape[1] == 1:
                idx = tlx.cast(tlx.reshape(tensor, (-1,)), tlx.int64)
                eye_np = np.eye(num_classes, dtype=np.float32)
                eye_tlx = tlx.convert_to_tensor(eye_np)
                return tlx.gather(eye_tlx, idx)
            return tlx.cast(tensor, tlx.float32)

        y_true = _to_one_hot(y_true, self.num_classes)
        y_pred = _to_one_hot(y_pred, self.num_classes)
        
        self.TP += tlx.reduce_sum(y_true * y_pred, axis=0)
        self.FP += tlx.reduce_sum((1 - y_true) * y_pred, axis=0)
        self.FN += tlx.reduce_sum(y_true * (1 - y_pred), axis=0)

    def compute(self, average: str=None):
        eps = 1e-10
        
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
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def get_cuda_mem(dev) -> float:
    return 0.0


def get_num_params(model: Module) -> float:
    num_paramst = sum([np.prod(param.shape) for param in model.trainable_weights])
    return num_paramst / 1e6


def get_mem_params(model: Module) -> float:
    mem_params = sum([np.prod(param.shape) * 4 for param in model.all_weights])
    return mem_params / (1024**2)