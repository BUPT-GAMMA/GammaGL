# !/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2022/04/16 08:31
# @Author  : clear
# @FileName: __init__.py


import os
if 'TL_BACKEND' in os.environ:
    if os.environ['TL_BACKEND'] == 'tensorflow':
        from .tensorflow import *

    elif os.environ['TL_BACKEND'] == 'mindspore':
        from .mindspore import *

    elif os.environ['TL_BACKEND'] == 'paddle':
        from .paddle import *

    elif os.environ['TL_BACKEND'] == 'torch':
        from .torch import *

    else:
        raise NotImplementedError("This backend is not supported")
else:
    from .tensorflow import *
