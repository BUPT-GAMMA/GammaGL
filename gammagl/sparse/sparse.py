# -*- coding: utf-8 -*-
# @author WuJing
# @created 2023/4/5

from ._sparse import c_random_walk
from gammagl.utils.platform_utils import ops_func


@ops_func
def random_walk(rowptr, col, start, walk_length):
    return c_random_walk(rowptr, col, start, walk_length)
