# -*- coding: utf-8 -*-
# @author WuJing
# @created 2023/3/22


from .tensor import unique as unique_impl


def boolean_dispatch(arg_name, arg_index, default, if_true, if_false, module_name, func_name):
    """
    Dispatches to either of 2 script functions based on a boolean argument.
    """

    def fn(*args, **kwargs):
        dispatch_flag = False
        if arg_name in kwargs:
            dispatch_flag = kwargs[arg_name]
        elif arg_index < len(args):
            dispatch_flag = args[arg_index]

        if dispatch_flag:
            return if_true(*args, **kwargs)
        else:
            return if_false(*args, **kwargs)

    return fn


def _unique_impl(input, sorted=True, return_inverse=False, return_counts=False):
    output, inverse, counts = unique_impl(input, sorted, return_inverse, return_counts)
    return output, inverse, counts


def _return_counts(input, sorted=True, return_inverse=False, return_counts=False):
    output, _, counts = _unique_impl(input, sorted, return_inverse, return_counts)
    return output, counts


def _return_output(input, sorted=True, return_inverse=False, return_counts=False):
    output, _, _ = _unique_impl(input, sorted, return_inverse, return_counts)
    return output


def _return_inverse(input, sorted=True, return_inverse=False, return_counts=False):
    output, inverse, _ = _unique_impl(input, sorted, return_inverse, return_counts)
    return output, inverse


_return_inverse_true = boolean_dispatch(
    arg_name='return_counts',
    arg_index=2,
    default=False,
    if_true=_unique_impl,
    if_false=_return_inverse,
    module_name=__name__,
    func_name='unique')

_return_inverse_false = boolean_dispatch(
    arg_name='return_counts',
    arg_index=2,
    default=False,
    if_true=_return_counts,
    if_false=_return_output,
    module_name=__name__,
    func_name='unique')

unique = boolean_dispatch(
    arg_name='return_inverse',
    arg_index=2,
    default=False,
    if_true=_return_inverse_true,
    if_false=_return_inverse_false,
    module_name=__name__,
    func_name='unique')
