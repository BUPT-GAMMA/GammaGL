import torch as th
import numpy as np
from gammagl.data import Graph


def random_aug(graph, attr, diag_attr, x, feat_drop_rate, edge_mask_rate, device):
    n_node = graph.num_nodes if hasattr(graph, 'num_nodes') else graph.x.shape[0]

    edge_mask = mask_edge(graph, edge_mask_rate)

    feat = drop_feature(x, feat_drop_rate)

    edge_index = graph.edge_index[:, edge_mask].long()  

    edge_weight = attr[edge_mask] if attr is not None else None

    if isinstance(attr, np.ndarray):
        attr_cpu = attr.cpu().numpy() if attr.is_cuda else attr.numpy()
        diag_attr_cpu = diag_attr.cpu().numpy() if diag_attr.is_cuda else diag_attr.numpy()
        attr = np.concatenate([attr_cpu[edge_mask], diag_attr_cpu], axis=0)

    new_graph = Graph(x=feat, edge_index=edge_index)

    new_graph.x = new_graph.x.to(device)
    new_graph.edge_index = new_graph.edge_index.to(device)

    if edge_weight is not None:
        edge_weight = th.tensor(edge_weight, dtype=th.float32).to(device)  # 确保 edge_weight 是 FloatTensor

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