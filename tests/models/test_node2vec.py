import os
os.environ['TL_BACKEND'] = 'torch'
import tensorlayerx as tlx
from gammagl.models import Node2vecModel


def test_node2vec():
    edge_index = tlx.convert_to_tensor([[0, 1, 1, 2], [1, 0, 2, 1]])

    model = Node2vecModel(edge_index=edge_index, embedding_dim=16, walk_length=2,
                          num_walks=1, p=1.0, q=1.0, edge_weight=None, window_size=2)

    assert model.campute()[0].shape == (3, 16)

    pos_rw = model.pos_sample()
    neg_rw = model.neg_sample()
    assert model.loss(pos_rw, neg_rw) >= 0

test_node2vec()
