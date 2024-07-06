from itertools import product
import scipy.sparse as sp
import os.path as osp
from typing import Callable, List, Optional
from gammagl.data.download import download_url
from gammagl.data import (HeteroGraph, InMemoryDataset, extract_zip)
import os
import numpy as np
import tensorlayerx as tlx
from gammagl.datasets.dblp import DBLP

def test_dblp():
    root = './temp'
    dataset = DBLP(root=root)
    data = dataset[0]
    assert 'author' in data.node_types, "Node type 'author' not found in data."
    assert 'paper' in data.node_types, "Node type 'paper' not found in data."
    assert 'term' in data.node_types, "Node type 'term' not found in data."
    assert 'conference' in data.node_types, "Node type 'conference' not found in data."
    assert data['author'].x.shape == (4057, 334), f"Author features shape mismatch: {data['author'].x.shape}"
    assert data['paper'].x.shape == (14328, 4231), f"Paper features shape mismatch: {data['paper'].x.shape}"
    assert data['term'].x.shape == (7723,), f"Term features shape mismatch: {data['term'].x.shape}"
    assert data['conference'].num_nodes == 20, f"Conference node count mismatch: {data['conference'].num_nodes}"
    assert data['author'].y.shape[0] == 4057, f"Author labels shape mismatch: {data['author'].y.shape}"
    assert data['author'].train_mask.sum().item() + data['author'].val_mask.sum().item() + data['author'].test_mask.sum().item() == 4057, "Train/Val/Test mask sum mismatch"
    print("All tests passed!")





