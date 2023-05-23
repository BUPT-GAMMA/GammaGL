import torch as th
import tensorlayerx as tlx
import numpy as np
from gammagl.data import Graph
from gammagl.utils import add_self_loops


def random_aug(graph, attr, diag_attr, x, feat_drop_rate, edge_mask_rate):
    edge_mask = mask_edge(graph, edge_mask_rate)  # 随机丢边
    feat = drop_feature(x, feat_drop_rate)  # 随机丢特征维度，将值设为0

    ng_edge_index = graph.edge_index[:, edge_mask]
    ng_edge_index = add_self_loops(edge_index=ng_edge_index, num_nodes=x.shape[0])[0]  # 加上自连边
    ng = Graph(edge_index=ng_edge_index)
    attr = th.cat([attr[edge_mask], diag_attr])  # 丢弃边上的weight，并把自连边的weight加上，防止丢弃自连边weight

    return ng, attr, feat


def drop_feature(x, drop_prob):
    drop_mask = th.empty(
        (x.size(1),),
        dtype=th.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x


def mask_edge(graph, mask_prob):
    E = graph.num_edges
    mask_rates = th.FloatTensor(np.ones(E) * mask_prob)
    masks = th.bernoulli(1 - mask_rates)
    mask_idx = masks.nonzero().squeeze(1)
    return mask_idx
