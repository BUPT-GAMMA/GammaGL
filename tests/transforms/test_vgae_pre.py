import numpy as np
import scipy.sparse as sp
from gammagl.transforms.vgae_pre import sparse_to_tuple, mask_test_edges


def test_vgae_pre():
    adj = sp.csr_matrix([
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0]
    ], dtype=np.float32)

    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)

    assert adj_train.shape == adj.shape, "Shape of adj_train does not match input adj"

    assert np.diag(adj_train.todense()).sum() == 0, "Diagonal elements are not removed properly"

    assert len(train_edges) + len(val_edges) + len(test_edges) == adj.nnz / 2, "Incorrect number of edges"
    assert len(val_edges_false) == len(val_edges), "Incorrect number of validation false edges"
    assert len(test_edges_false) == len(test_edges), "Incorrect number of test false edges"

    print("All tests passed!")
