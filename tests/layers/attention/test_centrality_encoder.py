
import tensorlayerx as tlx
from tensorlayerx import nn
from gammagl.utils import degree, mask_to_index
from gammagl.layers.attention.centrality_encoder import CentralityEncoding
def test_centrality_encoder():
    edge_index = tlx.convert_to_tensor([[0, 1, 2, 0, 3, 4, 4], [1, 0, 0, 2, 0, 3, 2]], dtype=tlx.int64)
    x = tlx.convert_to_tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0], [0.5, 0.5]], dtype=tlx.float32)
    max_in_degree = 3
    max_out_degree = 3
    node_dim = 2
    ce = CentralityEncoding(max_in_degree=max_in_degree, max_out_degree=max_out_degree, node_dim=node_dim)
    x_encoded = ce(x, edge_index)
    assert tlx.get_tensor_shape(x_encoded) == tlx.get_tensor_shape(x)
    assert not tlx.ops.is_nan(tlx.reduce_sum(x_encoded))


