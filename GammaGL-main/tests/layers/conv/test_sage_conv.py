import tensorlayerx as tlx
import numpy as np
from gammagl.layers.conv import SAGEConv


def test_sage_conv():
    x1 = np.random.uniform(low = 0, high = 1, size = (4, 8))
    x1 = tlx.convert_to_tensor(x1, dtype = tlx.float32)
    edge_index = tlx.convert_to_tensor([[0, 1, 2, 3], [0, 0, 1, 1]], dtype=tlx.int64)
    conv = SAGEConv(in_channels=8, out_channels=32)
    out = conv(x1, edge_index)
    assert tlx.get_tensor_shape(out) == [4, 32]

    # if is_full_test():
    #     t = '(Tensor, Tensor, Size) -> Tensor'
    #     jit = torch.jit.script(conv.jittable(t))
    #     assert jit(x1, edge_index).tolist() == out.tolist()
    #     assert jit(x1, edge_index, size=(4, 4)).tolist() == out.tolist()
    #
    #     t = '(Tensor, SparseTensor, Size) -> Tensor'
    #     jit = torch.jit.script(conv.jittable(t))
    #     assert jit(x1, adj.t()).tolist() == out.tolist()
    #
    # adj = adj.sparse_resize((4, 2))
    # conv = SAGEConv((8, 16), 32)
    # assert conv.__repr__() == 'SAGEConv((8, 16), 32)'
    # out1 = conv((x1, x2), edge_index)
    # out2 = conv((x1, None), edge_index, (4, 2))
    # assert out1.size() == (2, 32)
    # assert out2.size() == (2, 32)
    # assert conv((x1, x2), edge_index, (4, 2)).tolist() == out1.tolist()
    # assert conv((x1, x2), adj.t()).tolist() == out1.tolist()
    # assert conv((x1, None), adj.t()).tolist() == out2.tolist()
    #
    # if is_full_test():
    #     t = '(OptPairTensor, Tensor, Size) -> Tensor'
    #     jit = torch.jit.script(conv.jittable(t))
    #     assert jit((x1, x2), edge_index).tolist() == out1.tolist()
    #     assert jit((x1, x2), edge_index, size=(4, 2)).tolist() == out1.tolist()
    #     assert jit((x1, None), edge_index,
    #                size=(4, 2)).tolist() == out2.tolist()
    #
    #     t = '(OptPairTensor, SparseTensor, Size) -> Tensor'
    #     jit = torch.jit.script(conv.jittable(t))
    #     assert jit((x1, x2), adj.t()).tolist() == out1.tolist()
    #     assert jit((x1, None), adj.t()).tolist() == out2.tolist()
