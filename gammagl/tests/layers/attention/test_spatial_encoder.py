import tensorlayerx as tlx
from gammagl.layers.attention.spatial_encoder import SpatialEncoding


def test_spatial_encoder():
    encoder = SpatialEncoding(max_path_distance=5)
    x = tlx.random_normal(shape=(32, 64))
    paths = {
        0: {1: [0, 1, 2]},
        1: {2: [1, 2]},
        2: {}
    }
    spatial_matrix = encoder(x, paths)
    print(spatial_matrix)
    assert spatial_matrix.shape == (32, 32)
