
#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2022/04/16 08:31
# @Author  : clear
# @FileName: __init__.py


import os
import tensorlayerx as tlx
import numpy as np

def _get_tlx_backend():
    """Detect the actual backend used by TensorLayerX."""
    if 'TL_BACKEND' in os.environ:
        return os.environ['TL_BACKEND']
    
    t = tlx.zeros((1,))
    backend_name = type(t).__module__.lower()
    if 'tensorflow' in backend_name:
        return 'tensorflow'
    elif 'torch' in backend_name:
        return 'torch'
    elif 'paddle' in backend_name:
        return 'paddle'
    elif 'mindspore' in backend_name:
        return 'mindspore'
    elif 'jittor' in backend_name:
        return 'jittor'
    return 'tensorflow'

_backend = _get_tlx_backend()

if _backend == 'tensorflow':
    from .tensorflow import *

elif _backend == 'mindspore':
    from .mindspore import *

elif _backend == 'paddle':
    from .paddle import *

elif _backend == 'torch':
    from .torch import *

elif _backend == 'jittor':
    from .jittor import *

else:
    raise NotImplementedError(f"This backend is not supported: {_backend}")
