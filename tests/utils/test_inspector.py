import pytest
from gammagl.utils.inspector import Inspector


class Example:
    def method1(self, a: int, b: str) -> None:
        pass
    
    def method2(self, x: float) -> None:
        pass

def test_inspector():
    example = Example()
    inspector = Inspector(example)
    
    # Inspect the methods
    inspector.inspect(example.method1, pop_first=False)
    inspector.inspect(example.method2, pop_first=False)
    
    # Debugging information
    print("Inspector params after inspecting method1 and method2:")
    print(inspector.params)
    
    # Test keys method
    keys = inspector.keys()
    expected_keys = {"a", "b", "x"}
    assert keys == expected_keys, f"Unexpected keys: {keys}"
    
    # Test implements method
    assert inspector.implements("method1") is True, "method1 should be implemented"
    assert inspector.implements("method2") is True, "method2 should be implemented"
    
    # Test types method
    types = inspector.types()
    expected_types = {
        "a": "int",
        "b": "str",
        "x": "float"
    }
    assert types == expected_types, f"Unexpected types: {types}"
    
    # Test distribute method
    params = inspector.distribute("method1", {"a": 1, "b": "test"})
    expected_params = {"a": 1, "b": "test"}
    assert params == expected_params, f"Unexpected distributed params: {params}"
    
    params = inspector.distribute("method2", {"x": 2.5})
    expected_params = {"x": 2.5}
    assert params == expected_params, f"Unexpected distributed params: {params}"
