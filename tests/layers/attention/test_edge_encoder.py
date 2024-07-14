import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TL_BACKEND'] = 'tensorflow'
from gammagl.layers.attention.edge_encoder import EdgeEncoding
import tensorlayerx as tlx


def test_edge_encoder():
    edge_encoder = EdgeEncoding(edge_dim=3, max_path_distance=4)
    x = tlx.random_normal(shape=(5, 10))
    edge_attr = tlx.random_normal(shape=(10, 3))
    edge_paths = {
        0: {
            1: [0, 1, 2],
            2: [0, 3]
        },
        1: {
            2: [1, 2]
        },
        2: {}
    }
    cij = edge_encoder(x, edge_attr, edge_paths)
    assert cij.shape == (5, 5)


test_edge_encoder()