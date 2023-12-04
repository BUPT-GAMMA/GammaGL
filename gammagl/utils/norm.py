import tensorlayerx as tlx
from gammagl.mpops import *


def calc_gcn_norm(edge_index, num_nodes, edge_weight=None):
    """
    calculate GCN Normalization.

    Parameters
    ----------
    edge_index: 
        edge index
    num_nodes: 
        number of nodes of graph
    edge_weight: 
        edge weights of graph

    Returns
    -------
    tensor
        1-dim Tensor

    """
    src, dst = edge_index[0], edge_index[1]
    if edge_weight is None:
        edge_weight = tlx.ones(shape=(edge_index.shape[1], 1))  # torch backend `shape` should be tuple.
    deg = tlx.reshape(unsorted_segment_sum(edge_weight, src, num_segments=num_nodes), (-1,))
    deg_inv_sqrt = tlx.pow(deg, -0.5)
    weights = tlx.ops.gather(deg_inv_sqrt, src) * tlx.reshape(edge_weight, (-1,)) * tlx.ops.gather(deg_inv_sqrt, dst)
    return weights
