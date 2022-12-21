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
        node_idx = tlx.convert_to_tensor(np.array([node_idx]).flatten(), dtype=edge_index.dtype)
    
    subsets = [node_idx]
    node_mask = tlx.zeros((num_nodes,), dtype=tlx.bool)

    for _ in range(num_hops):
        node_mask = tlx.zeros((num_nodes,), dtype=tlx.bool)
        if tlx.BACKEND == 'paddle':
            node_mask[subsets[-1]] = True
        else:
            node_mask = tlx.scatter_update(node_mask, subsets[-1], tlx.ones_like(subsets[-1], dtype=tlx.bool))
        edge_mask = tlx.gather(node_mask, row)
        subsets.append(tlx.mask_select(col, edge_mask))
    
    subset, inv = np.unique(tlx.convert_to_numpy(tlx.concat(subsets, axis=0)), return_inverse=True)
    subset = tlx.convert_to_tensor(subset)
    inv = inv[:tlx.convert_to_numpy(tlx.count_nonzero(node_idx+1))]

    node_mask = tlx.zeros((num_nodes,), dtype=tlx.bool)
    if tlx.BACKEND == 'paddle':
        node_mask[subset] = True
    else:
        node_mask = tlx.scatter_update(node_mask, subset, tlx.ones_like(subset, dtype=tlx.bool))
    edge_mask = tlx.gather(node_mask, row) & tlx.gather(node_mask, col)

    edge_index = tlx.mask_select(edge_index, edge_mask, axis=1)

    if relabel_nodes:
        node_idx = tlx.constant(-1, dtype=node_idx.dtype, shape=(num_nodes,))
        node_idx = tlx.scatter_update(node_idx, subset, tlx.arange(0, subset.shape[0], dtype=edge_index.dtype))
        edge_index = tlx.gather(node_idx, edge_index)
    
    return subset, edge_index, inv, edge_mask