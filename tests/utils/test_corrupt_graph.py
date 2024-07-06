import numpy as np
from gammagl.data import Graph
import torch
from gammagl.utils.corrupt_graph import dfde_norm_g
import tensorlayerx as tlx
def test_dfde_norm_g():
    x = tlx.convert_to_tensor(np.array([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=np.float32))
    edge_index = tlx.convert_to_tensor(np.array([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=np.int64))
    feat_drop_rate = 0.5
    drop_edge_rate = 0.5
    new_g = dfde_norm_g(edge_index, x, feat_drop_rate, drop_edge_rate)
    assert new_g.edge_index.shape[0] == 2, "Edge index shape mismatch"
    assert new_g.x.shape[1] == x.shape[1], "Node feature shape mismatch"
    assert new_g.edge_weight.shape[0] == new_g.edge_index.shape[1], "Edge weight shape mismatch"




