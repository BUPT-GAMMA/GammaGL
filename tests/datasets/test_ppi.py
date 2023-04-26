import sys
import os
import tensorlayerx as tlx
from gammagl.datasets.ppi import PPI


root = './data'

def test_ppi_dataset():
    train_dataset = PPI(root)
    val_dataset = PPI(root, 'val')
    test_dataset = PPI(root, 'test')
    assert len(train_dataset) == 20
    assert len(val_dataset) == 2
    assert len(test_dataset) == 2

