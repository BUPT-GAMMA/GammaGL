import tensorlayerx as tlx
import numpy as np
from .num_nodes import maybe_num_nodes

def k_hop_subgraph(node_idx, num_hops, edge_index, relabel_nodes=False, num_nodes=None, reverse=False):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    if reverse:
        row, col = edge_index
    else:
        col, row = edge_index

    if isinstance(node_idx, (int, list, tuple)):
        node_idx = tlx.convert_to_tensor(np.array([node_idx]).flatten(), dtype=row.dtype, device=edge_index.device)
    else:
        node_idx = tlx.to_device(node_idx, edge_index.device)
    
    subsets = [node_idx]
    node_mask = tlx.zeros(num_nodes, dtype=tlx.bool, device=edge_index.device)

    for _ in range(num_hops):
        node_mask = tlx.zeros(num_nodes, dtype=tlx.bool, device=edge_index.device)
        node_mask = tlx.scatter_update(node_mask, subsets[-1], tlx.ones_like(subsets[-1], dtype=tlx.bool))
        edge_mask = tlx.gather(node_mask, row)
        subsets.append(tlx.mask_select(col, edge_mask))
    
    subset, inv = np.unique(tlx.concat(subsets, axis=0).numpy(), return_inverse=True)
    subset = tlx.convert_to_tensor(subset, device=edge_index.device)
    inv = inv[:tlx.count_nonzero(node_idx+1).numpy()]

    node_mask = tlx.zeros(num_nodes, dtype=tlx.bool, device=edge_index.device)
    node_mask = tlx.scatter_update(node_mask, subset, tlx.ones_like(subset, dtype=tlx.bool))
    edge_mask = tlx.gather(node_mask, row) & tlx.gather(node_mask, col)

    edge_index = tlx.mask_select(edge_index, edge_mask, axis=1)

    if relabel_nodes:
        node_idx = tlx.constant(-1, dtype=node_idx.dtype, shape=(num_nodes,), device=edge_index.device)
        node_idx = tlx.scatter_update(node_idx, subset, tlx.arange(subset.shape[0]))
        edge_index = tlx.gather(node_idx, edge_index)
    
    return subset, edge_index, inv, edge_mask