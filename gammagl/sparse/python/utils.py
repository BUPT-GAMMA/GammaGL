# -*- coding: utf-8 -*-
# @author WuJing
# @created 2023/4/4

import tensorlayerx as tlx


def assert_is_tensor_or_none(*args):
    for arg in args:
        assert tlx.is_tensor(arg) or arg is None
