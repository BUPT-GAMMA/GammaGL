import torch as th
import numpy as np
import dgl

def random_aug(graph, attr, diag_attr, x, feat_drop_rate, edge_mask_rate):
    n_node = graph.number_of_nodes()

    edge_mask = mask_edge(graph, edge_mask_rate)
    feat = drop_feature(x, feat_drop_rate)

    ng = dgl.graph([])
    ng.add_nodes(n_node)
    src = graph.edges()[0]
    dst = graph.edges()[1]
    nsrc = src[edge_mask]
    ndst = dst[edge_mask]
    ng.add_edges(nsrc, ndst)
    ng = ng.add_self_loop()
    
    attr = th.cat([attr[edge_mask], diag_attr])
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
    E = graph.number_of_edges()
    mask_rates = th.FloatTensor(np.ones(E) * mask_prob)
    masks = th.bernoulli(1 - mask_rates)
    mask_idx = masks.nonzero().squeeze(1)
    return mask_idx
evice)  # 确保 edge_weight 是 FloatTensor

    return new_graph, edge_weight, feat



def drop_feature(x, drop_prob):
    drop_mask = th.empty(
        (x.size(1),),
        dtype=th.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x


def mask_edge(graph, edge_mask_rate):
    E = graph.edge_index.shape[1]

    mask = np.random.rand(E) > edge_mask_rate

    return mask