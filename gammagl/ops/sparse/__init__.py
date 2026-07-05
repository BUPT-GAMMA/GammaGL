# -*- coding: utf-8 -*-
# @author WuJing
# @created 2023/4/11

try:
    from .sparse import (
        ind2ptr,
        ptr2ind,
        neighbor_sample,
        hetero_neighbor_sample,
        saint_subgraph,
        random_walk,
        sample_adj,
    )
except Exception:
    def _to_numpy(value):
        import numpy as np
        import tensorlayerx as tlx

        if isinstance(value, np.ndarray):
            return value
        return tlx.convert_to_numpy(value)

    def ind2ptr(ind, M, num_worker=0):
        import numpy as np
        import tensorlayerx as tlx

        ind_np = _to_numpy(ind).astype(np.int64, copy=False)
        counts = np.bincount(ind_np, minlength=M)
        ptr = np.concatenate(([0], np.cumsum(counts, dtype=np.int64)))
        return tlx.convert_to_tensor(ptr, dtype=tlx.int64)

    def ptr2ind(ptr, E, num_worker=1):
        import numpy as np
        import tensorlayerx as tlx

        ptr_np = _to_numpy(ptr).astype(np.int64, copy=False)
        counts = np.diff(ptr_np)
        ind = np.repeat(np.arange(len(counts), dtype=np.int64), counts)
        if E is not None:
            ind = ind[:E]
        return tlx.convert_to_tensor(ind, dtype=tlx.int64)

    def _missing_sparse_op(*args, **kwargs):
        raise ImportError(
            "GammaGL sparse C++ ops are not built. Reinstall GammaGL with "
            "native ops enabled to use this sampling operation."
        )

    neighbor_sample = _missing_sparse_op
    hetero_neighbor_sample = _missing_sparse_op
    saint_subgraph = _missing_sparse_op
    random_walk = _missing_sparse_op
    sample_adj = _missing_sparse_op
