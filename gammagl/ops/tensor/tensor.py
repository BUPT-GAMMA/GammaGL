# -*- coding: utf-8 -*-
# @author WuJing
# @created 2023/4/11

try:
    from ._unique import c_unique
except:
    from ._tensor import c_unique

from gammagl.utils.platform_utils import out_tensor_list


@out_tensor_list
def unique(input, sorted, return_inverse, return_counts):
    return c_unique(input, sorted, return_inverse, return_counts)
