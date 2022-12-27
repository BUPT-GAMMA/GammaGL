import tensorlayerx as tlx
from gammagl.layers.pool import global_sort_pool

def test_sort_pool():
    x = tlx.convert_to_tensor([[5.],[4.],[6.],[3.],[2.],[1.]])
    batch = tlx.convert_to_tensor([0,0,0,1,1,2])
    x = global_sort_pool(x, batch, 2)
    assert tlx.convert_to_numpy(x).tolist() == [[6., 5.], [3., 2.], [1., 0.]]
