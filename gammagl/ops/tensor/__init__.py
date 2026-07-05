# -*- coding: utf-8 -*-
# @author WuJing
# @created 2023/4/11

try:
    from .tensor import unique
except Exception:
    def unique(input, sorted=True, return_inverse=False, return_counts=False):
        import numpy as np
        import tensorlayerx as tlx

        array = tlx.convert_to_numpy(input)
        output, inverse, counts = np.unique(
            array,
            return_inverse=True,
            return_counts=True,
        )
        return (
            tlx.convert_to_tensor(output, dtype=input.dtype),
            tlx.convert_to_tensor(inverse, dtype=tlx.int64),
            tlx.convert_to_tensor(counts, dtype=tlx.int64),
        )
