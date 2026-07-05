import os
os.environ['TL_BACKEND'] = 'torch'
import tensorlayerx as tlx
from gammagl.layers.attention.defog_layer import XEyTransformerLayer

def test_defog_layer():
    dx, de, dy = 16, 8, 4
    n_head = 4
    layer = XEyTransformerLayer(dx, de, dy, n_head)

    bs, n = 2, 5
    X = tlx.ones((bs, n, dx))
    E = tlx.ones((bs, n, n, de))
    y = tlx.ones((bs, dy))
    node_mask = tlx.ones((bs, n))

    out_X, out_E, out_y = layer(X, E, y, node_mask)

    assert out_X.shape == (bs, n, dx)
    assert out_E.shape == (bs, n, n, de)
    assert out_y.shape == (bs, dy)

def test_defog_layer_edge_cases():
    dx, de, dy = 16, 8, 4
    n_head = 4
    layer = XEyTransformerLayer(dx, de, dy, n_head)

    # 1. Batch size 1
    bs, n = 1, 3
    X = tlx.ones((bs, n, dx))
    E = tlx.ones((bs, n, n, de))
    y = tlx.ones((bs, dy))
    node_mask = tlx.ones((bs, n))
    out_X, out_E, out_y = layer(X, E, y, node_mask)
    assert out_X.shape == (bs, n, dx)

    # 2. Empty graph (n=1)
    bs, n = 2, 1
    X = tlx.ones((bs, n, dx))
    E = tlx.ones((bs, n, n, de))
    y = tlx.ones((bs, dy))
    node_mask = tlx.ones((bs, n))
    out_X, out_E, out_y = layer(X, E, y, node_mask)
    assert out_X.shape == (bs, n, dx)

if __name__ == '__main__':
    test_defog_layer()
    test_defog_layer_edge_cases()
