import inspect
import re
from collections import OrderedDict
from itertools import product
from typing import Callable, Dict, List, Tuple

import pyparsing as pp
from gammagl.utils.typing import split_types_repr,sanitize,param_type_repr,return_type_repr,parse_types,resolve_types

def test_typing():
    #测试split_types_repr函数
    types_repr = "int, str, float"
    expected_output = ["int", "str", "float"]
    assert split_types_repr(types_repr) == expected_output
    # 嵌套类型
    types_repr = "Tuple[int, str]"
    expected_output = ["Tuple[int, str]"]
    assert split_types_repr(types_repr) == expected_output

    # 泛型类型
    types_repr = "List[Tuple[int, str], Union[float, NoneType]]"
    expected_output = ["List[Tuple[int, str]]", "Union[float, NoneType]"]
    print(split_types_repr(types_repr))
    assert split_types_repr(types_repr) == expected_output

    # 复杂嵌套类型
    types_repr = "Dict[str, List[Tuple[int, Union[float, NoneType]]]]"
    expected_output = ["Dict[str, List[Tuple[int, Union[float, NoneType]]]]"]
    assert split_types_repr(types_repr) == expected_output

    # 空类型
    types_repr = ""
    expected_output = [""]
    assert split_types_repr(types_repr) == expected_output


     #测试param_type_repr函数
    class MockParameter:
        annotation = float
    
    param = MockParameter()
    assert param_type_repr(param) == "float"

    # Mock parameter without annotation
    class MockParameter:
        annotation = inspect.Parameter.empty
    
    param = MockParameter()
    assert param_type_repr(param) == "torch.Tensor"
