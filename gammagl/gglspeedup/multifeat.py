import cupy as cp

from typing import List
import numpy as np

from .sharedfeat import CGPU_feat
from .utils import parse_size, reindex_feature


class Multi_CGPUFeature(object):
    def __init__(self,
                 rank: int,
                 device_list: List[int],
                 device_cache_size, csr=None
                 ):
        self.device_cache_size = device_cache_size

        self.device_list = device_list
        self.device_tensor_list = {}

        self.rank = rank
        self.csr = csr
        self.feature_order_cpu = None
        self.device_tensor_list = {}
        self.ipc_handle_ = None


    def cal_size(self, cpu_tensor, cache_memory_budget: int, data_size=4):
        element_size = cpu_tensor.shape[1] * data_size
        cache_size = cache_memory_budget // element_size
        return cache_size

    def partition(self, cpu_tensor, cache_memory_budget: int, data_size):

        cache_size = self.cal_size(cpu_tensor, cache_memory_budget, data_size)
        return [cpu_tensor[:cache_size], cpu_tensor[cache_size:]]

    def from_cpu_tensor(self, cpu_tensor, data_size=4):
        cache_memory_budget = parse_size(self.device_cache_size)
        print(
            f"LOG>>> {min(100, int(100 * cache_memory_budget / np.size(cpu_tensor) / data_size))}% data cached"
        )

        cpu_tensor, self.feature_order_cpu = reindex_feature(self.csr, cpu_tensor)

        cache_part, self.cpu_part = self.partition(cpu_tensor, cache_memory_budget, data_size)
        self.cpu_feat = self.cpu_part.copy()

        with cp.cuda.Device(self.rank):
            self.feature_order = cp.array(self.feature_order_cpu)

        for device in self.device_list:
            with cp.cuda.Device(device):
                cgpu_feat = CGPU_feat(device)
                self.device_tensor_list[device] = cgpu_feat
                if cache_part.shape[0] > 0:
                    self.device_tensor_list[device].add_gpu_feat(cache_part)


        # 构建CPU Tensor
        if np.size(self.cpu_part) > 0:
            self.device_tensor_list[self.rank].append(self.cpu_part)

    def __getitem__(self, node_idx):
        import torch
        if isinstance(node_idx, cp.ndarray):
            self.lazy_init_from_ipc_handle(node_idx.device.id)
            cur_device = node_idx.device.id
        elif isinstance(node_idx, torch.Tensor):
            self.lazy_init_from_ipc_handle(node_idx.device.index)
            cur_device = node_idx.device.index
        with cp.cuda.Device(cur_device):
            node_idx = self.feature_order[node_idx]
        return self.device_tensor_list[cur_device][node_idx]

    @property
    def ipc_handle(self):
        return self.ipc_handle_

    @ipc_handle.setter
    def ipc_handle(self, ipc_handle):
        self.ipc_handle_ = ipc_handle

    # 01
    def share_ipc(self):
        gpu_ipc_handle_dict = {}
        for device in self.device_tensor_list:
            gpu_ipc_handle_dict[device], self.cpu_part, ed = self.device_tensor_list[device].share_ipc()

        return gpu_ipc_handle_dict, self.cpu_feat if np.size(self.cpu_feat) > 0 else None, self.device_list, self.device_cache_size, self.feature_order_cpu, ed

    # 05
    def from_gpu_ipc_handle_dict(self, gpu_ipc_handle_dict, cpu_tensor, ed):

        ipc_handle = gpu_ipc_handle_dict[self.rank], cpu_tensor
        cgpu_feat = CGPU_feat.new_from_share_ipc(
                ipc_handle, self.rank)
        cgpu_feat.ed = ed
        self.device_tensor_list[self.rank] = cgpu_feat

        self.cpu_part = cpu_tensor
    ''''''

    # 03
    @classmethod
    def lazy_from_ipc_handle(cls, ipc_handle):
        gpu_ipc_handle_dict, cpu_part, device_list, device_cache_size, _, ed = ipc_handle
        feature = cls(device_list[0], device_list, device_cache_size)
        feature.ipc_handle = ipc_handle
        return feature

    # 04
    def lazy_init_from_ipc_handle(self, rank):
        if self.ipc_handle is None:
            return

        self.rank = rank
        gpu_ipc_handle_dict, cpu_part, device_list, device_cache_size, cpu_orders, ed = self.ipc_handle
        self.from_gpu_ipc_handle_dict(gpu_ipc_handle_dict, cpu_part, ed)

        with cp.cuda.Device(self.rank):
            self.feature_order = cp.array(cpu_orders)

        self.ipc_handle = None

    @property
    def shape(self):
        return self.device_tensor_list[self.rank].shape
