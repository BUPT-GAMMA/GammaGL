from itertools import product
import unittest
import scipy.sparse as sp
import os
import os.path as osp
from typing import Callable, List, Optional
import numpy as np
import tensorlayerx as tlx
from gammagl.data.download import download_url
from gammagl.data import (HeteroGraph, InMemoryDataset, extract_zip)
from gammagl.datasets.dblp import DBLP
def test_dblp():
    class TestDBLP(unittest.TestCase):
    
        def test_process(self):
            # Create an instance of DBLP with a temporary root directory
            dataset = DBLP(root='./temp')
            
            # Mock the download and extract operations (assuming files are already downloaded)
            # You can simulate this by placing the necessary files in the './temp' directory
            
            # Call the process method
            dataset.process()
            
            # Load the processed data
            data, slices = dataset.load_data(dataset.processed_paths[0])
            
            # Perform assertions to check if data is correctly processed
            self.assertIsInstance(data, HeteroGraph)
            
            # Example assertions based on your dataset structure
            # Check if nodes and edges are loaded correctly
            self.assertEqual(data['author'].x.shape, (4057, ...))  # Adjust ... based on your feature dimension
            self.assertEqual(data['author'].y.shape, (4057, ...))  # Adjust ... based on your label dimension
            self.assertEqual(data['paper'].edge_index.shape[1], 14328)  # Number of edges for papers
            
            # Check if masks are correctly set
            self.assertEqual(data['author'].train_mask.sum().item(), ...)  # Adjust ... based on your train mask count
            self.assertEqual(data['author'].val_mask.sum().item(), ...)    # Adjust ... based on your val mask count
            self.assertEqual(data['author'].test_mask.sum().item(), ...)   # Adjust ... based on your test mask count
            
            # Additional assertions based on your specific data structure and processing
            
            # Example: self.assertTrue(transformed_data.some_property)