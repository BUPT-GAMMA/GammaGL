import tensorlayerx as tlx
import numpy as np
from gammagl.mpops import unsorted_segment_sum
from gammagl.utils import calc_gcn_norm


def test_calc_gcn_norm():
  
    edge_index = np.array([
        [0, 1, 1, 2, 2, 3],
        [1, 0, 2, 1, 3, 2]
    ])
    num_nodes = 4
    weights = calc_gcn_norm(tlx.convert_to_tensor(edge_index), num_nodes)
    degree = np.array([1, 2, 2, 1])
    deg_inv_sqrt = np.power(degree, -0.5)
    expected_weights = np.array([
        deg_inv_sqrt[0] * deg_inv_sqrt[1],
        deg_inv_sqrt[1] * deg_inv_sqrt[0],
        deg_inv_sqrt[1] * deg_inv_sqrt[2],
        deg_inv_sqrt[2] * deg_inv_sqrt[1],
        deg_inv_sqrt[2] * deg_inv_sqrt[3],
        deg_inv_sqrt[3] * deg_inv_sqrt[2]
    ])
    weights_np = tlx.convert_to_numpy(weights)
    assert np.allclose(weights_np, expected_weights)


