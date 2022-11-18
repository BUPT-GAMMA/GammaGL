import tensorlayerx as tlx
from gammagl.utils import to_dense_batch

def test_to_dense_batch():
    tlx.set_device()
    x = tlx.convert_to_tensor([[1,2], [3,4], [5,6], [6,7]])
    batch = tlx.convert_to_tensor([0, 1, 1, 1])
    a, b = to_dense_batch(x, batch)
    assert tlx.convert_to_numpy(a).tolist() == [[[1,2],[0,0],[0,0]],[[3,4],[5,6],[6,7]]]
    assert tlx.convert_to_numpy(b).tolist() == [[True, False, False], [True, True, True]]