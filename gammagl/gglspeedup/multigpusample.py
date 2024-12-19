from typing import List

import cupy as cp
import scipy.sparse as sp
import numpy as np
import time
import ThrustRTC as trtc
import CURandRTC as rndrtc
from dataclasses import dataclass
import cupy as cp

from .cupy_and_thrust_code.kernels import *



@dataclass(frozen=True)
class _FakeDevice(object):
    pass


class SAP_Kernel:
    def __init__(self, device, csr_adj):
        self.indptr = None
        self.indices = None
        self.device = device
        with cp.cuda.Device(device):
            self.indptr = cp.array(csr_adj.indptr, dtype=cp.int64)
            self.indices = cp.array(csr_adj.indices, dtype=cp.int64)
        self.indptr_th = trtc.DVCupyVector(self.indptr)
        self.indices_th = trtc.DVCupyVector(self.indices)

    def process_and_sample(self, dst_node, K):
        if K > 0:
            with cp.cuda.Device(self.device):
                out = cp.empty_like(dst_node)
                outdegs = cp.empty_like(dst_node)
                cnt_ptr = cp.empty_like(dst_node)
                dst_th = trtc.DVCupyVector(dst_node)
                out_th = trtc.DVCupyVector(out)
                outdegs_th = trtc.DVCupyVector(outdegs)
                sample_number = trtc.DVInt64(K)
                cnt_ptr_th = trtc.DVCupyVector(cnt_ptr)
                perprocess0.launch_n(dst_node.shape[0], [dst_th, self.indptr_th, out_th, outdegs_th, sample_number])
                trtc.Exclusive_Scan(out_th, cnt_ptr_th)
                tot = trtc.Reduce(out_th)
                sample_node_idx = cp.empty(shape=(tot,), dtype=cp.int64)
                sample_node_idx_th = trtc.DVCupyVector(sample_node_idx)
                rng = rndrtc.DVRNG()
                sample_kernel0.launch_n(dst_node.shape[0], [rng, dst_th, outdegs_th, out_th, cnt_ptr_th,
                                                            self.indptr_th, self.indices_th, sample_node_idx_th])
                sample_node_idx = sample_node_idx.copy()
            return out, cnt_ptr, sample_node_idx, tot
        else:
            with cp.cuda.Device(self.device):
                out = cp.empty_like(dst_node)
                cnt_ptr = cp.empty_like(out)
                dst_th = trtc.DVCupyVector(dst_node)
                out_th = trtc.DVCupyVector(out)
                cnt_ptr_th = trtc.DVCupyVector(cnt_ptr)
                perprocess1.launch_n(dst_node.shape[0], [dst_th, self.indptr_th, out_th])
                trtc.Exclusive_Scan(out_th, cnt_ptr_th)
                tot = trtc.Reduce(out_th)
                sample_node_idx = cp.empty(shape=(tot,), dtype=cp.int64)
                sample_node_idx_th = trtc.DVCupyVector(sample_node_idx)

                sample_kernel1.launch_n(dst_node.shape[0], [dst_th, out_th, cnt_ptr_th,
                                                            self.indptr_th, self.indices_th, sample_node_idx_th])
            return out, cnt_ptr, sample_node_idx, tot

    def sample_and_reindex(self, dst_node, K):
        cnt, cntptr, sample_nodes, _ = self.process_and_sample(dst_node, K)
        with cp.cuda.Device(self.device):
            all_nodes = cp.concatenate((dst_node, sample_nodes), axis=0)
            keys = cp.full((all_nodes.shape[0] * 2,), fill_value=-1, dtype=cp.int64)
            values = cp.full((all_nodes.shape[0] * 2,), fill_value=999999999, dtype=cp.int64)
            pre_reindex_kernel1((all_nodes.shape[0],), (1,), (all_nodes, keys, values, keys.shape[0]))

            locs = cp.zeros((all_nodes.shape[0] + 1,), dtype=cp.int64)

            pre_reindex_kernel2((all_nodes.shape[0],), (1,), (all_nodes, keys, values, locs, keys.shape[0]))
            locs_th = trtc.DVCupyVector(locs)
            tot = trtc.Reduce(locs_th)
            unique_id = cp.empty((tot,), dtype=cp.int64)
            trtc.Exclusive_Scan(locs_th, locs_th)

            unique_code_kernel((all_nodes.shape[0],), (1,), (all_nodes, keys, values, locs, unique_id, keys.shape[0]))
            rowreindex_kernel((sample_nodes.shape[0],), (1,), (sample_nodes, keys, values, locs, keys.shape[0]))
            row = sample_nodes
            col = cp.empty_like(row)
            colreindex_kernel((dst_node.shape[0],), (1,), (cp.arange(dst_node.shape[0]), cnt, cntptr, col))
        return unique_id, row, col, (unique_id.shape[0], dst_node.shape[0])

    def sample(self, dst_node, sizes, device):
        # 这里处理的是cpu的tensor
        with cp.cuda.Device(self.device):
            dst_node = cp.array(dst_node)
        items = []
        for size in sizes:
            dst_node, row, col, size = self.sample_and_reindex(dst_node, size)
            items.append(Adj(row, col, size))
        return dst_node, items[::-1]


class MultiGPUSampler:
    def __init__(self,
                 device, csr=None):

        self.cs = None
        self.csr = csr
        if device is not _FakeDevice:
            self.cs = SAP_Kernel(device, csr)

        self.device = device

    def set_size(self,sizes):
        self.sizes = sizes
    def sample(self, dst, rank):
        self.lazy_init(rank)
        # print("device:", rank)
        # print(self.indptrs[rank].device)
        return self.cs.sample(dst, self.sizes, rank)

    def lazy_init(self, device):
        if self.cs is not None:
            return
        # use torch to make sure which device is used
        self.cs = SAP_Kernel(device, self.csr)

    def share_ipc(self):

        return self.csr

    @classmethod
    def lazy_from_ipc_handle(cls, ipc_handle):

        csr = ipc_handle
        return cls(_FakeDevice, csr)
