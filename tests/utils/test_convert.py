import numpy as np
import tensorlayerx as tlx
import networkx as nx
import scipy.sparse as ssp
from gammagl.utils.convert import to_scipy_sparse_matrix, to_networkx
from gammagl.data import Graph

def test_convert():
    edge_index = tlx.convert_to_tensor([
        [0, 1, 1, 2, 2, 3],
        [1, 0, 2, 1, 3, 2],
    ])
    edge_attr = tlx.convert_to_tensor([0.5, 0.5, 0.2, 0.2, 0.1, 0.1])
    num_nodes = 4
    scipy_matrix = to_scipy_sparse_matrix(edge_index, edge_attr, num_nodes)
    expected_scipy_matrix = ssp.coo_matrix((
        [0.5, 0.5, 0.2, 0.2, 0.1, 0.1],
        ([0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2])
    ), shape=(4, 4))
    assert np.allclose(scipy_matrix.toarray(), expected_scipy_matrix.toarray())

    edge_index = tlx.convert_to_tensor([
        [0, 1, 1, 2, 2, 3],
        [1, 0, 2, 1, 3, 2],
    ])
    num_nodes = 4
    data = Graph(edge_index=edge_index, num_nodes=num_nodes)
    G = to_networkx(data)
    G_type = type(G)
    expected_G = G_type()
    expected_edges = [(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)]
    expected_G.add_edges_from(expected_edges)
    assert nx.is_isomorphic(G, expected_G)
