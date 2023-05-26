import tensorlayerx as tlx
import numpy as np
from gammagl.data import Graph
from gammagl.utils import add_self_loops


def random_aug(graph, attr, diag_attr, x, feat_drop_rate, edge_mask_rate):
    edge_mask = mask_edge(graph, edge_mask_rate)  # randomly drop edges
    feat = drop_feature(x, feat_drop_rate)  # randomly drop the feature dimension and set the value to 0

    ng_edge_index = graph.edge_index[:, edge_mask]
    ng_edge_index = add_self_loops(edge_index=ng_edge_index, num_nodes=x.shape[0])[0]  # add self-loop
    ng = Graph(edge_index=ng_edge_index)
    attr = tlx.concat([attr[edge_mask], diag_attr], axis=-1)  # add self-loop weight

    return ng, attr, feat


def drop_feature(x, drop_prob):
    mask_value = np.random.uniform(low=0, high=1, size=x.size(1))
    mask = mask_value < drop_prob  # generate saving mask
    x = x.clone()
    x[:, mask] = 0
    return x


def mask_edge(graph, mask_prob):
    E = graph.num_edges
    masks = np.random.binomial(n=1, p=1 - mask_prob, size=E)
    masks_idx = list(np.nonzero(masks)[0])  # get saved edges' index
    return masks_idx
