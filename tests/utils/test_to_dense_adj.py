import tensorlayerx as tlx
from gammagl.mpops import unsorted_segment_sum
from gammagl.utils.to_dense_adj import to_dense_adj  
import numpy as np

def test_to_dense_adj():
    edge_index = tlx.convert_to_tensor([
        [0, 1, 3],
        [1, 2, 4]
    ], dtype=tlx.int64)

    batch = tlx.convert_to_tensor([0, 0, 0, 1, 1], dtype=tlx.int64)

    adj_matrix = to_dense_adj(edge_index, batch=batch)

    adj_matrix_np = tlx.convert_to_numpy(adj_matrix)

    expected_output = tlx.convert_to_tensor([
        [
            [
                [0, 1, 0],
                [0, 0, 1],
                [0, 0, 0]
            ],
            [
                [0, 1],
                [0, 0]
            ]
        ]
    ], dtype=tlx.float32)

    assert np.array_equal(adj_matrix_np, tlx.convert_to_numpy(expected_output)), "The test failed, adjacency matrices do not match."
