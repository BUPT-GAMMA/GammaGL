# -*- coding: utf-8 -*-
# @author WuJing
# @created 2023/4/11

import numpy as np

try:
    import torch
except Exception:
    torch = None

try:
    from .sparse import (
    ind2ptr,
    ptr2ind,
    neighbor_sample,
    hetero_neighbor_sample,
    saint_subgraph,
    random_walk,
    sample_adj
)
except:
    def ind2ptr(ind, M, num_worker=0):
        """Python fallback for CSR ptr construction."""
        if torch is not None and isinstance(ind, torch.Tensor):
            ind = ind.to(dtype=torch.long)
            ptr = torch.zeros(int(M) + 1, dtype=torch.long, device=ind.device)
            if ind.numel() > 0:
                ones = torch.ones_like(ind, dtype=torch.long)
                ptr.scatter_add_(0, ind + 1, ones)
            return torch.cumsum(ptr, dim=0)

        ind_np = np.asarray(ind, dtype=np.int64).reshape(-1)
        ptr = np.zeros(int(M) + 1, dtype=np.int64)
        if ind_np.size > 0:
            np.add.at(ptr, ind_np + 1, 1)
        return np.cumsum(ptr)

    def ptr2ind(ptr, E=None, num_worker=0):
        """Python fallback for CSR ptr to indices."""
        if torch is not None and isinstance(ptr, torch.Tensor):
            ptr = ptr.to(dtype=torch.long)
            if E is None:
                E = int(ptr[-1].item())
            counts = ptr[1:] - ptr[:-1]
            rows = torch.arange(int(ptr.shape[0]) - 1, device=ptr.device, dtype=torch.long)
            return torch.repeat_interleave(rows, counts)[:int(E)]

        ptr_np = np.asarray(ptr, dtype=np.int64).reshape(-1)
        if E is None:
            E = int(ptr_np[-1]) if ptr_np.size > 0 else 0
        counts = ptr_np[1:] - ptr_np[:-1]
        rows = np.arange(ptr_np.shape[0] - 1, dtype=np.int64)
        return np.repeat(rows, counts)[:int(E)]

    def _missing_sparse_ext(*args, **kwargs):
        raise RuntimeError(
            "gammagl sparse C++ ops are unavailable. "
            "Please compile extensions to use this API."
        )

    neighbor_sample = _missing_sparse_ext
    hetero_neighbor_sample = _missing_sparse_ext
    saint_subgraph = _missing_sparse_ext
    random_walk = _missing_sparse_ext
    sample_adj = _missing_sparse_ext
