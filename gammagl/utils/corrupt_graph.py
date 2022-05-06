import tensorlayerx as tlx
# import tensorflow as tf
import scipy.sparse as sp
from gammagl.data import Graph
import numpy as np

# calculate D^(-1/2) * A_hat * D^(-1/2)
def calc(edge, num_node):
    weight = np.ones(edge.shape[1])
    sparse_adj = sp.coo_matrix((weight, (edge[0], edge[1])), shape=(num_node, num_node))
    A = (sparse_adj + sp.eye(num_node)).tocoo()
    col, row, weight = A.col, A.row, A.data
    deg = np.array(A.sum(1))
    deg_inv_sqrt = np.power(deg, -0.5).flatten()

    return col, row, np.array(deg_inv_sqrt[row] * weight * deg_inv_sqrt[col], dtype=np.float32)

# drop feat and drop edge to create new graph
def dfde_norm_g(edge_index, feat, feat_drop_rate, drop_edge_rate):
    num_node = feat.shape[0]
    edge_mask = drop_edge(edge_index, drop_edge_rate)
    edge_index = edge_index.numpy()
    new_edge = edge_index.T[edge_mask]
    # tlx can't assignment, so still use tf
    feat = drop_feat(feat, feat_drop_rate)
    row, col, weight = calc(new_edge.T, num_node)
    new_g = Graph(edge_index=tlx.convert_to_tensor([row, col], dtype=tlx.int64), x=feat, num_nodes=num_node)
    new_g.edge_weight = tlx.convert_to_tensor(weight)

    return new_g

# random drop edge to generate mask
def drop_edge(edge_index, drop_edge_rate=0.5):
    r"""Randomly drops edges from the adjacency matrix
        :obj:`(edge_index, edge_attr)` with probability :obj:`p` using samples from
        a Bernoulli distribution.

        Args:
        edge_index (Tensor): The edge indices.
        p (float, optional): Dropout probability. (default: :obj:`0.5`)
        """
    if drop_edge_rate < 0. or drop_edge_rate > 1.:
        raise ValueError(f'Dropout probability has to be between 0 and 1 '
                         f'(got {drop_edge_rate}')
    if drop_edge_rate == 0.:
        return tlx.convert_to_tensor(np.ones(edge_index.shape[1], dtype=np.bool8))
    mask = tlx.ops.random_uniform(shape=[edge_index.shape[1]],
                                  minval=0, maxval=1, dtype=tlx.float32) >= drop_edge_rate
    return mask

# random drop node's feat
def drop_feat(feat, drop_feat_rate):
    if drop_feat_rate < 0. or drop_feat_rate >= 1.:
        raise ValueError(f'Dropout probability has to be between 0 and 1 '
                         f'(got {drop_feat_rate}')
    if drop_feat_rate == 0.:
        drop_mask = np.ones(feat.shape[1], dtype=np.bool8)
    else:
        drop_mask = np.random.uniform(size=(feat.shape[1], )) < drop_feat_rate
    # use numpy
    feat = feat.numpy()
    feat[:, drop_mask] = 0

    return tlx.convert_to_tensor(feat)

