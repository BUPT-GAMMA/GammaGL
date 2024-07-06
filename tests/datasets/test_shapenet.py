import json
import os
import os.path as osp
import shutil
import unittest

import tensorlayerx as tlx
from gammagl.data import (
    Graph,
    InMemoryDataset,
    download_url,
    extract_zip,
)
from gammagl.io import read_txt_array
from gammagl.datasets.shapenet import ShapeNet


import numpy as np

import os
import tensorlayerx as tlx
from gammagl.datasets.shapenet import ShapeNet

def test_shapenet():
    root = './temp'
    dataset = ShapeNet(root=root, categories='Airplane', include_normals=True)
    data = dataset[0]
    assert data.pos.shape[1] == 3, "Position shape mismatch"
    assert data.x is not None, "Node features should not be None"
    assert data.x.shape[1] == 3, "Node feature shape mismatch"
    assert data.y is not None, "Labels should not be None"
    print("All tests passed!")




