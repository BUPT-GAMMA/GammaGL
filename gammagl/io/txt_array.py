import numpy as np


def parse_txt_array(src, sep=None, start=0, end=None, dtype=None, device=None):
    to_number = int
    if dtype == np.float32:
        to_number = float
    src = [[to_number(x) for x in line.split(sep)[start:end]] for line in src]
    src = np.array(src, dtype=dtype).squeeze()
    return src


def read_txt_array(path, sep=None, start=0, end=None, dtype=None, device=None):
    with open(path, 'r') as f:
        src = f.read().split('\n')[:-1]
    return parse_txt_array(src, sep, start, end, dtype, device)
