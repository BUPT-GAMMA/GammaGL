
import tensorlayerx as tlx
from gammagl.mpops import unsorted_segment_sum
from gammagl.utils.to_dense_adj import to_dense_adj  
import numpy as np

def test_to_dense_adj():
    edge_index = tlx.convert_to_tensor([
        [0, 1, 2, 3, 1],
        [1, 0, 3, 2, 2]
    ], dtype=tlx.int64)
    batch = tlx.convert_to_tensor([0, 0, 1, 1], dtype=tlx.int64)
    expected_output = tlx.convert_to_tensor([
        [
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ],
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0]
        ]
    ], dtype=tlx.float32)
    adj = to_dense_adj(edge_index, batch)
    result = tlx.convert_to_numpy(adj)
    assert np.array_equal(result, tlx.convert_to_numpy(expected_output))
    # edge_index = tlx.convert_to_tensor([
    #     [0, 1],
    #     [1, 0]
    # ], dtype=tlx.int64)

    # batch = tlx.convert_to_tensor([0, 0], dtype=tlx.int64)

    # expected_output = tlx.convert_to_tensor([
    #     [
    #         [0, 1],
    #         [1, 0]
    #     ]
    # ], dtype=tlx.int64)

    # adj = to_dense_adj(edge_index, batch)
    # result = tlx.convert_to_numpy(adj)

    # assert np.array_equal(result, tlx.convert_to_numpy(expected_output)), f"Expected {expected_output}, but got {result}"


    # edge_index = tlx.convert_to_tensor([
    #     [0, 1, 2],
    #     [1, 0, 2]
    # ], dtype=tlx.int64)

    # batch = tlx.convert_to_tensor([0, 0, 0], dtype=tlx.int64)

    # expected_output = tlx.convert_to_tensor([
    #     [
    #         [0, 1, 0],
    #         [1, 0, 0],
    #         [0, 0, 1]
    #     ]
    # ], dtype=tlx.int64)

    # adj = to_dense_adj(edge_index, batch)
    # result = tlx.convert_to_numpy(adj)

    # assert np.array_equal(result, tlx.convert_to_numpy(expected_output)), f"Expected {expected_output}, but got {result}"


    # edge_index = tlx.convert_to_tensor([
    #     [0, 1, 2, 3],
    #     [1, 0, 3, 2]
    # ], dtype=tlx.int64)

    # batch = tlx.convert_to_tensor([0, 0, 1, 1], dtype=tlx.int64)
    # edge_attr = tlx.convert_to_tensor([0.5, 0.5, 1.0, 1.0], dtype=tlx.float32)

    # expected_output = tlx.convert_to_tensor([
    #     [
    #         [0, 0.5],
    #         [0.5, 0],
    #         [0, 0],
    #         [0, 0]
    #     ],
    #     [
    #         [0, 0],
    #         [0, 0],
    #         [0, 1.0],
    #         [1.0, 0]
    #     ]
    # ], dtype=tlx.float32)

    # adj = to_dense_adj(edge_index, batch, edge_attr)
    # result = tlx.convert_to_numpy(adj)

    # assert np.allclose(result, tlx.convert_to_numpy(expected_output)), f"Expected {expected_output}, but got {result}"

   
