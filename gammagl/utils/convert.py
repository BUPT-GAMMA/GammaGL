import scipy.sparse as ssp
import tensorlayerx as tlx
import numpy as np

from .num_nodes import maybe_num_nodes


def to_scipy_sparse_matrix(edge_index, edge_attr = None, num_nodes = None):
    r"""Converts a graph given by edge indices and edge attributes to a scipy
    sparse matrix.
    Args:
        edge_index: The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)
    """
    row, col = tlx.convert_to_numpy(edge_index)
    
    if edge_attr is None:
        edge_attr = np.ones(row.shape[0])
    else:
        edge_attr = tlx.convert_to_numpy(tlx.reshape(edge_attr, (-1,)))
        assert edge_attr.shape[0] == row.shape[0]
    
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    return ssp.coo_matrix((edge_attr, (row, col)), (num_nodes, num_nodes))