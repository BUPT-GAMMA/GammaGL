import tensorlayerx as tlx
from gammagl.utils import add_self_loops
from gammagl.mpops import unsorted_segment_sum

def accuracy_tlx(logits, labels):
    if tlx.BACKEND == 'tensorflow':
        predicted_labels = tlx.argmax(logits, axis=-1)
        correct = tlx.reduce_sum(tlx.cast(tlx.equal(predicted_labels, labels), dtype=tlx.float32))
        return correct / tlx.cast(tlx.get_tensor_shape(labels)[0], dtype=tlx.float32)
    elif tlx.BACKEND == 'torch':
        _, indices = tlx.ops.max(logits, dim=1)
        correct = tlx.reduce_sum(tlx.cast(indices == labels, dtype=tlx.float32))
        return correct / float(len(labels))
    elif tlx.BACKEND == 'paddle':
        predicted_labels = tlx.argmax(logits, axis=-1)
        correct = tlx.reduce_sum(tlx.cast(tlx.equal(predicted_labels, labels), dtype=tlx.float32))
        return correct / float(tlx.get_tensor_shape(labels)[0])
    elif tlx.BACKEND == 'mindspore':
        predicted_labels = tlx.argmax(logits, axis=-1)
        correct = tlx.reduce_sum(tlx.cast(tlx.equal(predicted_labels, labels), dtype=tlx.float32))
        return correct / float(tlx.get_tensor_shape(labels)[0])
    else:
        raise NotImplementedError(f"Accuracy not implemented for backend: {tlx.BACKEND}")


def compute_gcn_norm(edge_index, num_nodes, dtype, add_self_loops_flag=True):
    """
    Computes GCN normalization for edges (D^-0.5 * A * D^-0.5).
    Adapted from PyG's GCNConv normalization.

    Args:
        edge_index: Edge index tensor.
        num_nodes: Number of nodes in the graph.
        dtype: Dtype for the edge weights.
        add_self_loops_flag: Whether to add self-loops before normalization.

    Returns:
        Tuple[Tensor, Tensor]: Normalized edge_index and edge_weight.
    """
    edge_src, edge_dst = edge_index[0], edge_index[1]

    if add_self_loops_flag:
        edge_index_sl, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        edge_src_sl, edge_dst_sl = edge_index_sl[0], edge_index_sl[1]
    else:
        edge_index_sl = edge_index
        edge_src_sl, edge_dst_sl = edge_src, edge_dst
    
    edge_weight_ones = tlx.ones(shape=(tlx.get_tensor_shape(edge_src_sl)[0],), dtype=dtype)
    
    # Calculate D_out (degree of source nodes of the SL-augmented graph)
    deg_out = unsorted_segment_sum(edge_weight_ones, edge_src_sl, num_segments=num_nodes)
    # Calculate D_in (degree of destination nodes of the SL-augmented graph)
    deg_in = unsorted_segment_sum(edge_weight_ones, edge_dst_sl, num_segments=num_nodes)

    # Inverse square root, handling degree 0
    deg_out_inv_sqrt = tlx.pow(deg_out, -0.5)
    # deg_out_inv_sqrt = tlx.where(tlx.is_finite(deg_out_inv_sqrt), deg_out_inv_sqrt, tlx.zeros_like(deg_out_inv_sqrt))
    deg_in_inv_sqrt = tlx.pow(deg_in, -0.5)
    # deg_in_inv_sqrt = tlx.where(tlx.is_finite(deg_in_inv_sqrt), deg_in_inv_sqrt, tlx.zeros_like(deg_in_inv_sqrt))
    
    # For deg_out_inv_sqrt
    is_inf_deg_out = tlx.is_inf(deg_out_inv_sqrt)
    is_nan_deg_out = tlx.is_nan(deg_out_inv_sqrt)

    deg_out_inv_sqrt_finite_mask_tlx = tlx.logical_and(
        tlx.logical_not(is_inf_deg_out),
        tlx.logical_not(is_nan_deg_out)
    )

    # For deg_in_inv_sqrt
    is_inf_deg_in = tlx.is_inf(deg_in_inv_sqrt)
    is_nan_deg_in = tlx.is_nan(deg_in_inv_sqrt)
    deg_in_inv_sqrt_finite_mask_tlx = tlx.logical_and(
        tlx.logical_not(is_inf_deg_in),
        tlx.logical_not(is_nan_deg_in)
    )
    
    deg_out_inv_sqrt = tlx.where(deg_out_inv_sqrt_finite_mask_tlx, deg_out_inv_sqrt, tlx.zeros_like(deg_out_inv_sqrt))
    deg_in_inv_sqrt = tlx.where(deg_in_inv_sqrt_finite_mask_tlx, deg_in_inv_sqrt, tlx.zeros_like(deg_in_inv_sqrt))
    
    # Normalized edge weights: deg_out_inv_sqrt[src] * deg_in_inv_sqrt[dst]
    # (For A_ij, it's D_ii^-0.5 * A_ij * D_jj^-0.5)
    # If edge is (j,i), then weight is deg(j)^-0.5 * deg(i)^-0.5
    norm_edge_weight = tlx.gather(deg_out_inv_sqrt, edge_src_sl) * tlx.gather(deg_in_inv_sqrt, edge_dst_sl)
    
    return edge_index_sl, norm_edge_weight