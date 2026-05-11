# -*- coding: utf-8 -*-
# @author WuJing
# @created 2023/4/11

import numpy as np

try:
    import torch
except Exception:
    torch = None

try:
    from .tensor import unique  # C++ extension path
except Exception:
    def unique(input, sorted=True, return_inverse=False, return_counts=False):
        """Python fallback when tensor C++ extension is unavailable."""
        if torch is not None and isinstance(input, torch.Tensor):
            return torch.unique(
                input,
                sorted=sorted,
                return_inverse=return_inverse,
                return_counts=return_counts,
            )

        arr = np.asarray(input)
        out = np.unique(arr, return_inverse=return_inverse, return_counts=return_counts)
        if return_inverse and return_counts:
            values, inverse, counts = out
            return values, inverse, counts
        if return_inverse and not return_counts:
            values, inverse = out
            return values, inverse
        if (not return_inverse) and return_counts:
            values, counts = out
            return values, counts
        return out
