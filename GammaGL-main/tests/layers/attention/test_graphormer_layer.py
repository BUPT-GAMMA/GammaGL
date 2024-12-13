import tensorlayerx as tlx
from gammagl.layers.attention.edge_encoder import EdgeEncoding
from gammagl.layers.attention.graphormer_layer import GraphormerLayer


def test_graphormer_layer():
    if tlx.BACKEND == "tensorflow":
        return
    layer = GraphormerLayer(node_dim=64, edge_dim=16, n_heads=8, max_path_distance=5)
    x = tlx.random_normal(shape=(32, 64))  
    edge_attr = tlx.random_normal(shape=(64, 16)) 
    b = tlx.random_normal(shape=(32, 32)) 
    edge_paths = {
        0: {1: [0, 1, 2]},  
        1: {2: [1, 2]},    
        2: {}              
    }
    ptr = None  
    output = layer(x, edge_attr, b, edge_paths, ptr)
    assert output.shape == (32, 64)

    
