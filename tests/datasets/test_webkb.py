import os.path as osp
import os
import numpy as np
import tensorlayerx as tlx
from gammagl.utils import coalesce
from gammagl.data import InMemoryDataset, download_url, Graph
from gammagl.datasets.webkb import WebKB
def test_webkb():
    root = './temp'
    dataset = WebKB(root=root, name='Cornell')
    data = dataset[0]
    print(data)
    assert data.x.shape[0] > 0, "Node features should not be empty"
    assert data.edge_index.shape[0] == 2, "Edge index shape mismatch"
    assert data.y.shape[0] == data.x.shape[0], "Labels shape mismatch"
    print(data.train_mask.shape[0])
    print(data.val_mask.shape[0])
    print(data.test_mask.shape[0])
    print(data.x.shape)
    assert data.train_mask.shape[0] == data.x.shape[0], "Train mask shape mismatch"
    assert data.val_mask.shape[0] == data.x.shape[0], "Validation mask shape mismatch"
    assert data.test_mask.shape[0] == data.x.shape[0], "Test mask shape mismatch"
    print("All tests passed!")
