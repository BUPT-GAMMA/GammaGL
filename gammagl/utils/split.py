import tensorlayerx as tlx
import numpy as np

def split(y):

    train_ratio = 0.1
    val_ratio = 0.1
    test_ratio = 0.8

    N = len(y)
    train_num = int(N * train_ratio)
    val_num = int(N * (train_ratio + val_ratio))

    idx = np.arange(N)
    np.random.shuffle(idx)

    train_idx = idx[:train_num]
    val_idx = idx[train_num:val_num]
    test_idx = idx[val_num:]

    train_idx = tlx.convert_to_tensor(train_idx)
    val_idx = tlx.convert_to_tensor(val_idx)
    test_idx = tlx.convert_to_tensor(test_idx)

    return train_idx, val_idx, test_idx
