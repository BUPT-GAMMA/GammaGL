import tensorlayerx as tlx
from gammagl.data import Graph
import copy
import numpy as np

from gammagl.utils import add_self_loops, calc_gcn_norm


def dfde_norm_g(edge_index, feat, feat_drop_rate, drop_edge_rate):
    n_node = int(tlx.reduce_max(edge_index[0]) + 1)
    if isinstance(edge_index, np.ndarray):
        edge_mask = mask_edge(edge_index, drop_edge_rate)
    else:
        edge_index = tlx.convert_to_numpy(edge_index)
        edge_mask = mask_edge(edge_index, drop_edge_rate)
    if isinstance(feat, np.ndarray):
        feat = drop_feature(feat, feat_drop_rate)
    else:
        feat = drop_feature(tlx.convert_to_numpy(feat), feat_drop_rate)

    src = edge_index[0]
    dst = edge_index[1]

    nsrc = tlx.convert_to_tensor(src[edge_mask])
    ndst = tlx.convert_to_tensor(dst[edge_mask])
    edge_index = tlx.stack([nsrc, ndst])
    edge_index, _ = add_self_loops(edge_index)
    weight = calc_gcn_norm(edge_index, n_node)
    new_g = Graph(edge_index=edge_index, x=feat, num_nodes=n_node)
    new_g.edge_weight = weight

    return new_g


def drop_feature(x, drop_feat_rate):
    if drop_feat_rate < 0. or drop_feat_rate >= 1.:
        raise ValueError(f'Dropout probability has to be between 0 and 1 '
                         f'(got {drop_feat_rate}')
    drop_mask = np.random.uniform(0, 1, size=(x.shape[1])) < drop_feat_rate
    x = copy.copy(x)
    x[:, drop_mask] = 0

    return x


def mask_edge(edge_index, mask_prob):
    E = edge_index.shape[1]
    mask = np.random.uniform(0, 1, size=(E,)) >= mask_prob

    return mask
