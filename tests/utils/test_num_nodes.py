import numpy as np
import torch
from gammagl.utils.num_nodes import maybe_num_nodes, maybe_num_nodes_dict
from copy import copy
from gammagl.utils.check import check_is_numpy
import tensorlayerx as tlx

def test_num_nodes():
    edge_index_dict = {
        ('social', 'user-user'): np.array([[0, 1, 1, 2], [1, 0, 2, 1]]),
        ('knowledge', 'concept-concept'): np.array([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    }
    edge_index_tensor = torch.tensor(edge_index_dict[('social', 'user-user')]) 
    num_nodes = maybe_num_nodes(edge_index_tensor) 
    assert num_nodes == 3
    num_nodes_dict = maybe_num_nodes_dict(edge_index_dict)  
    expected_num_nodes_dict = {'social': 3, 'knowledge': 4}
    assert num_nodes_dict == expected_num_nodes_dict
