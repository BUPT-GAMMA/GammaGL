import tensorlayerx as tlx
import numpy as np
from .num_nodes import maybe_num_nodes

def k_hop_subgraph(node_idx, num_hops, edge_index, relabel_nodes=False, num_nodes=None, reverse=False):
    r"""Computes the induced subgraph of :obj:`edge_index` around all nodes in
    :attr:`node_idx` reachable within :math:`k` hops.
    The :attr:`flow` argument denotes the direction of edges for finding
    :math:`k`-hop neighbors. If set to :obj:`"source_to_target"`, then the
    method will find all neighbors that point to the initial set of seed nodes
    in :attr:`node_idx.`
    This mimics the natural flow of message passing in Graph Neural Networks.
    The method returns (1) the nodes involved in the subgraph, (2) the filtered
    :obj:`edge_index` connectivity, (3) the mapping from node indices in
    :obj:`node_idx` to their new location, and (4) the edge mask indicating
    which edges were preserved.
    Args:
        node_idx (int, list, tuple or Tensor): The central seed
            node(s).
        num_hops (int): The number of hops :math:`k`.
        edge_index : The edge indices.
        relabel_nodes (bool, optional): If set to :obj:`True`, the resulting
            :obj:`edge_index` will be relabeled to hold consecutive indices
            starting from zero. (default: :obj:`False`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        reverse (bool, optional): The flow direction of :math:`k`-hop, :obj:`False` for "source to target" or vice versa.
    """
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
        node_mask = tlx.zeros((num_nodes,), dtype = tlx.int64)
        node_mask = tlx.scatter_update(node_mask, subsets[-1], tlx.ones_like(subsets[-1], dtype = tlx.int64))
        edge_mask = tlx.gather(node_mask, row)
        edge_mask = tlx.cast(edge_mask, dtype = tlx.bool)
        subsets.append(tlx.mask_select(col, edge_mask))
    subset, inv = np.unique(tlx.convert_to_numpy(tlx.concat(subsets, axis=0)), return_inverse=True)
    subset = tlx.convert_to_tensor(subset)
    if tlx.BACKEND == 'paddle':
        idx = tlx.convert_to_numpy(tlx.count_nonzero(node_idx+1))[0]
    else:
        idx = tlx.convert_to_numpy(tlx.count_nonzero(node_idx+1))
    inv = inv[:idx]
        
    node_mask = tlx.zeros((num_nodes,), dtype = tlx.int64)
    node_mask = tlx.scatter_update(node_mask, subset, tlx.ones_like(subset, dtype = tlx.int64))
    if tlx.BACKEND != 'paddle':
        node_mask = tlx.cast(node_mask, dtype = tlx.bool)
    edge_mask = tlx.logical_and(tlx.gather(node_mask, row), tlx.gather(node_mask, col))
    edge_mask = tlx.cast(edge_mask, dtype = tlx.bool)

    edge_index = tlx.mask_select(edge_index, edge_mask, axis=1)
    if relabel_nodes:
        node_idx = tlx.constant(-1, dtype=node_idx.dtype, shape=(num_nodes,))
        node_idx = tlx.scatter_update(node_idx, subset, tlx.arange(0, subset.shape[0], dtype=edge_index.dtype))
        if tlx.BACKEND == 'paddle':
            edge_index = tlx.gather(node_idx, tlx.reshape(edge_index, (-1, 1)))
            edge_index = tlx.reshape(edge_index, (2, -1))
        else:
            edge_index = tlx.gather(node_idx, edge_index)
    return subset, edge_index, inv, edge_mask