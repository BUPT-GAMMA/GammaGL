import numpy as np


def check_is_numpy(*data):
    """
    Check if the given datas have numpy.array
    """
    for d in data:
        if isinstance(d, np.ndarray):
            return True
    return False
