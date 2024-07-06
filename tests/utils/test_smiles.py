from typing import List, Dict, Any
import tensorlayerx as tlx
from rdkit import Chem, RDLogger
from gammagl.data import Graph
from gammagl.utils.smiles import from_smiles


def test_from_smiles():
    smiles = 'CCO'  
    graph = from_smiles(smiles)
    expected_node_features = [
        [6, 0, 4, 5, 3, 0, 4, 0, 0], 
        [6, 0, 4, 5, 2, 0, 4, 0, 0],  
        [8, 0, 2, 5, 1, 0, 4, 0, 0], 
    ]
    expected_node_features = tlx.convert_to_tensor(expected_node_features, dtype=tlx.int64)
    expected_edge_indices = [
        [0, 1], [1, 0], 
        [1, 2], [2, 1], 
    ]
    expected_edge_indices = tlx.transpose(tlx.convert_to_tensor(expected_edge_indices))
    expected_edge_features = [
        [1, 0, 0], [1, 0, 0],  
        [1, 0, 0], [1, 0, 0], 
    ]
    expected_edge_features = tlx.convert_to_tensor(expected_edge_features, dtype=tlx.int64)
    assert tlx.convert_to_numpy(graph.x).tolist() == tlx.convert_to_numpy(expected_node_features).tolist()  
    assert tlx.convert_to_numpy(graph.edge_index).tolist() == tlx.convert_to_numpy(expected_edge_indices).tolist()  
    assert tlx.convert_to_numpy(graph.edge_attr).tolist() == tlx.convert_to_numpy(expected_edge_features).tolist()
