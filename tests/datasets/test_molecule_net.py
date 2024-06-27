import os
import os.path as osp
import re
from typing import Callable, Dict, Optional, Tuple, Union

from gammagl.data import download_url
from gammagl.data import InMemoryDataset
from gammagl.data.extract import extract_gz
import tensorlayerx as tlx

from gammagl.utils.smiles import from_smiles

import unittest

from gammagl.datasets.molecule_net import MoleculeNet  # Replace with the correct module path

class TestMoleculeNet(unittest.TestCase):

    def setUp(self):
        # Set up the dataset for testing
        self.dataset = MoleculeNet(root='./temp', name='esol')

    def test_download(self):
        # Test if the download method works correctly
        self.dataset.download()
        raw_file = os.path.join(self.dataset.raw_dir, self.dataset.raw_file_names)
        self.assertTrue(os.path.exists(raw_file))

    def test_process(self):
        # Mock the download method since we cannot actually download during testing
        self.dataset.download = lambda: None

        # Create some mock data for testing
        mock_data = """\
SMILES,Solubility
CCO,1.2
CCC,0.5
"""
        with open(os.path.join(self.dataset.raw_dir, self.dataset.raw_file_names), 'w') as f:
            f.write(mock_data)

        # Process the mock data
        self.dataset.process()

        # Load the processed data
        data, slices = self.dataset.load_data(self.dataset.processed_paths[0])

        # Check if data is correctly processed
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0].smiles, 'CCO')
        self.assertAlmostEqual(data[0].y.item(), 1.2)
        self.assertEqual(data[1].smiles, 'CCC')
        self.assertAlmostEqual(data[1].y.item(), 0.5)

    def test_repr(self):
        # Test the __repr__ method
        self.assertEqual(repr(self.dataset), 'ESOL(1128)')  # Adjust the expected value based on your dataset

