import numpy as np
import scipy.sparse as sp
import tensorlayerx as tlx
from gammagl.utils.loop import remove_self_loops
from gammagl.utils.undirected import to_undirected
from gammagl.data import Graph


def read_npz(path):
    with np.load(path) as f:
        return parse_npz(f)


def parse_npz(data):
    x = sp.csr_matrix((data['attr_data'], data['attr_indices'], data['attr_indptr']),
                      data['attr_shape']).todense()
    x = np.array(x)
    x[x > 0] = 1
    x = tlx.convert_to_tensor(x,dtype=tlx.float32)

    adj = sp.csr_matrix((data['adj_data'], data['adj_indices'], data['adj_indptr']),
                        data['adj_shape']).tocoo()
    edge_index = np.array([adj.row, adj.col])
    edge_index, _ = remove_self_loops(edge_index)
    edge_index = tlx.convert_to_tensor(edge_index)

    edge_index = to_undirected(edge_index, num_nodes=x.shape[0])

    y = tlx.convert_to_tensor(data['labels'])

    return Graph(x=x, edge_index=edge_index, y=y)
