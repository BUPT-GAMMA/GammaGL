import numpy as np
import tensorlayerx as tlx
from gammagl.layers.conv import MGNNI_m_iter


def test_mgnni_m_iter():
    m = 10
    k = 2
    threshold = 1e-5
    max_iter = 50
    gamma = 0.5
    num_nodes = 10
    edge_index = tlx.convert_to_tensor(np.array([
        [0, 1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5, 6]
    ]), dtype=tlx.int64)
    X = tlx.convert_to_tensor(np.random.randn(num_nodes, m), dtype=tlx.float32)
    edge_weight = tlx.convert_to_tensor(np.random.rand(edge_index.shape[1], 1), dtype=tlx.float32)
    conv = MGNNI_m_iter(m=m, k=k, threshold=threshold, max_iter=max_iter, gamma=gamma)
    try:
        out = conv(X, edge_index=edge_index, edge_weight=edge_weight, num_nodes=num_nodes)
        assert out.shape == (num_nodes, m)
    except Exception as e:
        assert False, f"运行时出错: {e}"
