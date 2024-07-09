import tensorlayerx as tlx
from gammagl.models import SpaSpeNode, Encoder, EigenMLP
import numpy as np
from scipy.sparse import csr_matrix
import scipy.sparse.linalg

def test_spaspenode():
    if tlx.BACKEND == "tensorflow":
        return
    edge_index = tlx.convert_to_tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=tlx.int64)
    x = tlx.convert_to_tensor(np.random.randn(3, 5), dtype=tlx.float32)
    y = tlx.convert_to_tensor([0, 1, 2], dtype=tlx.int64)

    num_nodes = x.shape[0]
    row, col = edge_index
    data_adj = csr_matrix((np.ones(len(row)), (row, col)), shape=(num_nodes, num_nodes))
    degree = np.array(data_adj.sum(axis=1)).flatten()
    deg_inv_sqrt = 1.0 / np.sqrt(degree)
    deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0
    I = csr_matrix(np.eye(num_nodes))
    D_inv_sqrt = csr_matrix((deg_inv_sqrt, (np.arange(num_nodes), np.arange(num_nodes))))
    L = I - D_inv_sqrt.dot(data_adj).dot(D_inv_sqrt)

    k = min(2, num_nodes - 1)
    e, u = scipy.sparse.linalg.eigsh(L, k=k, which='SM', tol=1e-3)
    e = tlx.convert_to_tensor(e, dtype=tlx.float32)
    u = tlx.convert_to_tensor(u, dtype=tlx.float32)

    model = SpaSpeNode(input_dim=x.shape[1], spe_dim=20, hidden_dim=32, output_dim=16, period=20)


    h_node_spa, h_node_spe = model(x, edge_index, e, u)
    assert h_node_spa.shape == (num_nodes, 16)
    assert h_node_spe.shape == (num_nodes, 16)
