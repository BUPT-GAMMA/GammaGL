import os
os.environ['TL_BACKEND'] = 'torch'
import tensorlayerx as tlx
from gammagl.models.defog import DeFoGModel
import torch

def test_defog_model_shape_and_mask():
    input_dims = {'X': 5, 'E': 4, 'y': 2}
    hidden_mlp_dims = {'X': 16, 'E': 8, 'y': 16}
    hidden_dims = {
        'dx': 16, 'de': 8, 'dy': 16,
        'n_head': 2, 'dim_ffX': 32, 'dim_ffE': 16, 'dim_ffy': 32
    }
    output_dims = {'X': 5, 'E': 4, 'y': 0}
    
    model = DeFoGModel(
        n_layers=1,
        input_dims=input_dims,
        hidden_mlp_dims=hidden_mlp_dims,
        hidden_dims=hidden_dims,
        output_dims=output_dims
    )
    
    bs, n = 2, 5
    X = tlx.ones((bs, n, input_dims['X']))
    E = tlx.ones((bs, n, n, input_dims['E']))
    # Test symmetric input E
    E = (E + tlx.transpose(E, perm=(0, 2, 1, 3))) / 2.0
    
    y = tlx.ones((bs, input_dims['y']))
    
    # Test node_mask (one node masked out in the second graph)
    node_mask = tlx.ones((bs, n))
    node_mask_tensor = tlx.convert_to_tensor(node_mask)
    if isinstance(node_mask_tensor, torch.Tensor):
        node_mask_tensor[1, 4] = 0.0
    
    out_X, out_E, out_y = model(X, E, y, node_mask_tensor)
    
    assert out_X.shape == (bs, n, output_dims['X'])
    assert out_E.shape == (bs, n, n, output_dims['E'])
    assert out_y.shape == (bs, 0)
    
    # Test symmetric output E
    out_E_transpose = tlx.transpose(out_E, perm=(0, 2, 1, 3))
    # We allow some precision error
    diff = tlx.abs(out_E - out_E_transpose)
    if isinstance(diff, torch.Tensor):
        assert torch.max(diff).item() < 1e-4

def test_defog_model_edge_cases():
    input_dims = {'X': 5, 'E': 4, 'y': 2}
    hidden_mlp_dims = {'X': 16, 'E': 8, 'y': 16}
    hidden_dims = {'dx': 16, 'de': 8, 'dy': 16, 'n_head': 2, 'dim_ffX': 32, 'dim_ffE': 16, 'dim_ffy': 32}
    output_dims = {'X': 5, 'E': 4, 'y': 2}
    
    model = DeFoGModel(1, input_dims, hidden_mlp_dims, hidden_dims, output_dims)
    
    # 1. Batch size 1
    bs, n = 1, 3
    X = tlx.ones((bs, n, input_dims['X']))
    E = tlx.ones((bs, n, n, input_dims['E']))
    y = tlx.ones((bs, input_dims['y']))
    node_mask = tlx.ones((bs, n))
    out_X, out_E, out_y = model(X, E, y, node_mask)
    assert out_X.shape == (1, 3, 5)
    
    # 2. Empty graph (n=1, technically a single node graph, n=0 may crash torch layers)
    bs, n = 2, 1
    X = tlx.ones((bs, n, input_dims['X']))
    E = tlx.ones((bs, n, n, input_dims['E']))
    y = tlx.ones((bs, input_dims['y']))
    node_mask = tlx.ones((bs, n))
    out_X, out_E, out_y = model(X, E, y, node_mask)
    assert out_X.shape == (2, 1, 5)

if __name__ == '__main__':
    test_defog_model_shape_and_mask()
    test_defog_model_edge_cases()
    print("DeFoGModel advanced tests passed!")
