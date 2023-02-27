import tensorlayerx as tlx
from gammagl.models import DeepWalkModel


def test_deepwalk():
    edge_index = tlx.convert_to_tensor([[0, 1, 1, 2], [1, 0, 2, 1]])

    model = DeepWalkModel(edge_index=edge_index, embedding_dim=16, walk_length=2,
                          num_walks=1, edge_weight=None, window_size=2)

    assert tlx.get_tensor_shape(model.campute()[0]) == [3, 16]

    pos_rw = model.pos_sample()
    neg_rw = model.neg_sample()
    assert model.loss(pos_rw, neg_rw) >= 0
