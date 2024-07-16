import numpy as np
from gammagl.utils.check import check_is_numpy


def test_check():
    assert check_is_numpy(np.array([1, 2, 3])) == True 
    assert check_is_numpy([1, 2, 3]) == False
    assert check_is_numpy([1, 2, 3], np.array([1, 2, 3]), "string") == True
    assert check_is_numpy([1, 2, 3], "string", 42) == False
    assert check_is_numpy() == False
    assert check_is_numpy(np.array(1)) == True
    assert check_is_numpy([[1, 2, 3], [4, 5, 6]]) == False
    assert check_is_numpy([1, 2, 3], np.array([4, 5, 6]), {'key': 'value'}, 42) == True
    assert check_is_numpy(None) == False
    assert check_is_numpy(np.array([[1, 2, 3], [4, 5, 6]])) == True
