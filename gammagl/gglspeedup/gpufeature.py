import cupy as cp

from typing import List

import numpy as np

from .utils import parse_size, reindex_feature

from .sharedfeat import CGPU_feat


class CGPUFeature(object):

    def __init__(self, csr,
                 device: int,
                 device_cache_size=0):

        self.device_cache_size = device_cache_size

        self.device = device
        # device{"cuda:0":tensor, "cuda:1":tensor, ...}
        self.device_tensor = None
        # self.default_device = device_list[0]
        self.feature_order = None
        self.csr = csr


    # 3
    def cal_size(self, cpu_tensor, cache_memory_budget: int, data_size):
        element_size = cpu_tensor.shape[1] * data_size
        cache_size = cache_memory_budget // element_size
        return cache_size

    # 2
    def partition(self, cpu_tensor, cache_memory_budget: int, data_size):

        cache_size = self.cal_size(cpu_tensor, cache_memory_budget, data_size)
        return [cpu_tensor[:cache_size], cpu_tensor[cache_size:]]

    # 1
    def from_cpu_tensor(self, cpu_tensor, data_size=4):

        cache_memory_budget = parse_size(self.device_cache_size)
        # feat is float32, defalut=4

        print(
            f"LOG>>> {min(100, int(100 * cache_memory_budget / np.size(cpu_tensor) / data_size))}% data cached"
        )

        cpu_tensor, feature_order = reindex_feature(self.csr, cpu_tensor)

        cache_part, self.cpu_part = self.partition(cpu_tensor, cache_memory_budget, data_size)
        #if isinstance(self.cpu_part, tf.Tensor):
        #    self.cpu_part = tf.raw_ops.Copy(self.cpu_part)
        #self.cpu_part = self.cpu_part.clone()


        with cp.cuda.Device(self.device):
            self.feature_order = cp.array(feature_order)
            cgpu_feat = CGPU_feat(self.device)
            self.device_tensor = cgpu_feat
            if cache_part.shape[0] > 0:
                self.device_tensor.add_gpu_feat(cache_part)


        # 构建CPU Tensor
        if np.size(self.cpu_part) > 0:
            self.device_tensor.append(self.cpu_part)

    # 4
    def __getitem__(self, node_idx):
        with cp.cuda.Device(self.device):
            node_idx = self.feature_order[node_idx]
            shard_tensor = self.device_tensor
            return shard_tensor[node_idx]


    def dim(self):
        return len(self.shape)
    @property
    def shape(self):
        return self.device_tensor.shape





