
from numba import cuda
import cupy as cp
import numpy as np


# 获取feat的kernel，不能直接切片，直接切片速度会非常感人
@cuda.jit(device=True)
def get_single_feat(out_feat, feat, dim):
    for i in range(dim):
        out_feat[i] = feat[i]


@cuda.jit()
def get_feat(map_feat, torch_feat, all_nodes, out_feat, ind, ed):
    tx = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x
    ty = cuda.blockDim.y * cuda.blockIdx.y + cuda.threadIdx.y
    if all_nodes[tx] < ed:
        out_feat[ind[tx]][ty] = torch_feat[all_nodes[tx]][ty]
    else:
        out_feat[ind[tx]][ty] = map_feat[all_nodes[tx] - ed][ty]


@cuda.jit()
def get_uva_feat(map_feat, all_nodes, out_feat, ind):
    tx = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x
    ty = cuda.blockDim.y * cuda.blockIdx.y + cuda.threadIdx.y

    out_feat[ind[tx]][ty] = map_feat[all_nodes[tx]][ty]


class CGPU_feat:
    def __init__(self, device):
        """
        put feat to GPU, must put it first, and confirm the GPU device_id
        :param feat:
        :param device:
        """
        self.device = device
        self.gpu_part = None
        self.cpu_part = None
        self.ed = None

    def add_gpu_feat(self, feat):
        self.ed = feat.shape[0]
        total_size = np.size(feat) * 4
        print(
            f"LOG >>> Memory Budge On {self.device} is {total_size // 1024 // 1024} MB"
        )
        with cp.cuda.Device(self.device):
            self.gpu_part = cp.array(feat)

    def append(self, cpu_feat):

        with cuda.gpus[self.device]:
            self.cpu_part = cuda.mapped_array(cpu_feat.shape, dtype=np.float32)
            self.cpu_part[:, :] = cpu_feat[:, :]

    def __getitem__(self, node_idx):
        aa = node_idx.shape[0]
        if self.cpu_part is None:
            with cp.cuda.Device(self.device):
                return self.gpu_part[node_idx]
        elif self.gpu_part is None:
            with cp.cuda.Device(self.device):
                out_feat = cp.empty((aa, self.cpu_part.shape[1]), dtype=cp.float32)
                ind = cp.arange(node_idx.shape[0], dtype=cp.int64)
            with cuda.gpus[self.device]:
                get_uva_feat[(aa, 1), (1, self.cpu_part.shape[1])](self.cpu_part, node_idx, out_feat, ind)
            del ind
            return out_feat
        else:
            with cp.cuda.Device(self.device):
                out_feat = cp.empty((aa, self.cpu_part.shape[1]), dtype=cp.float32)
                ind = cp.arange(node_idx.shape[0], dtype=cp.int64)
            with cuda.gpus[self.device]:
                get_feat[(aa, 1), (1, self.cpu_part.shape[1])](self.cpu_part, self.gpu_part, node_idx, out_feat, ind,
                                                               self.ed)
            del ind
            return out_feat

    @property
    def shape(self):
        if self.cpu_part is not None and self.gpu_part is not None:
            return (self.gpu_part.shape[0] + self.cpu_part.shape[0], self.cpu_part.shape[1])
        elif self.cpu_part is not None:
            return (self.cpu_part.shape[0], self.cpu_part.shape[1])
        else:
            return (self.gpu_part.shape[0], self.gpu_part.shape[1])
    # 02
    def share_ipc(self):
        handler = None
        if self.gpu_part is not None:
            with cp.cuda.Device(self.device):
                handler = cp.cuda.runtime.ipcGetMemHandle(self.gpu_part.data.ptr), self.gpu_part.shape, self.gpu_part.dtype
        return handler, self.cpu_part, self.ed

    # 07
    def from_ipc_handle(self, gpu_ipc, cpu_feat):
        if gpu_ipc is not None:
            with cp.cuda.Device(self.device):
                data_ptr = cp.cuda.runtime.ipcOpenMemHandle(gpu_ipc[0])
                buff = {"data": (int(data_ptr), False), "shape": gpu_ipc[1], "typestr": gpu_ipc[2]}
                class capsule(object):
                    pass
                cap = capsule()
                cap.__cuda_array_interface__ = buff
                self.gpu_part = cp.asanyarray(cap)

        if cpu_feat is not None:
            with cuda.gpus[self.device]:
                self.cpu_part = cuda.mapped_array(cpu_feat.shape, dtype=np.float32)
            self.cpu_part[:, :] = cpu_feat[:, :]


    # 06
    @classmethod
    def new_from_share_ipc(cls, ipc_handles, current_device):
        gpu_part_ipc, cpu_tensor = ipc_handles
        cgpu_feat = cls(current_device)
        cgpu_feat.from_ipc_handle(gpu_part_ipc, cpu_tensor)
        return cgpu_feat

