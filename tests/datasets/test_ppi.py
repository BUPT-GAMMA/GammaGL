import sys
import os
import tensorlayerx as tlx
sys.path.insert(0, os.path.abspath('../../'))  # add system file path.
from gammagl.datasets.ppi import PPI

tlx.set_device("GPU", 7)

root = '../../../data'

def test_ppi_dataset():
    train_dataset = PPI(root)
    val_dataset = PPI(root, 'val')
    test_dataset = PPI(root, 'test')
    assert len(train_dataset) == 20
    assert len(val_dataset) == 2
    assert len(test_dataset) == 2

test_ppi_dataset()