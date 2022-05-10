import numpy as np
import scipy.sparse as sp


def calc_sign(graph, feat):

    col, row = graph.edge_index.numpy()
    deg = np.array(graph.csr_adj.sum(1))
    deg_inv_sqrt = np.power(deg, -0.5).flatten()

    weight = np.ones_like(col)
    new_weight = deg_inv_sqrt[row] * weight * deg_inv_sqrt[col]
    new_adj = sp.coo_matrix((new_weight, [col, row]))
    return new_adj @ feat

