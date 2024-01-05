import tensorlayerx as tlx
from gammagl.mpops import unsorted_segment_sum

def to_dense_adj(
    edge_index,
    batch = None,
    edge_attr = None,
    max_num_nodes = None,
):

    if batch is None:
        num_nodes = int(tlx.reduce_max(edge_index)) + 1 if tlx.numel(edge_index) > 0 else 0
        batch = tlx.zeros([num_nodes], dtype=tlx.int64)

    batch_size = int(tlx.reduce_max(batch)) + 1 if tlx.numel(batch) > 0 else 1
    one = tlx.ones(shape=(batch.size(0),), dtype=tlx.float32)
    num_nodes = tlx.cast(unsorted_segment_sum(one, batch, batch_size), dtype=tlx.int64)
    cum_nodes = tlx.concat([tlx.zeros([1],dtype=tlx.int64), tlx.cumsum(num_nodes)])

    idx0 = batch[edge_index[0]]
    idx1 = edge_index[0] - cum_nodes[batch][edge_index[0]]
    idx2 = edge_index[1] - cum_nodes[batch][edge_index[1]]

    if max_num_nodes is None:
        max_num_nodes = int(tlx.reduce_max(num_nodes))
    elif ((tlx.numel(idx1) > 0 and tlx.reduce_max(idx1) >= max_num_nodes)
          or (tlx.numel(idx2) > 0 and tlx.reduce_max(idx2) >= max_num_nodes)):
        mask = (idx1 < max_num_nodes) & (idx2 < max_num_nodes)
        idx0 = tlx.mask_select(idx0, mask)
        idx1 = tlx.mask_select(idx1, mask)
        idx2 = tlx.mask_select(idx2, mask)
        edge_attr = None if edge_attr is None else tlx.mask_select(edge_attr, mask)

    if edge_attr is None:
        edge_attr = tlx.ones(tlx.numel(idx0), dtype=tlx.int64)

    size = [batch_size, max_num_nodes, max_num_nodes]
    size += tlx.get_tensor_shape(edge_attr)[1: ]
    adj = tlx.zeros(size, dtype=tlx.int64)
    flattened_size = batch_size * max_num_nodes * max_num_nodes
    adj = tlx.reshape(adj, [flattened_size] + tlx.get_tensor_shape(adj)[3: ])
    idx = idx0 * max_num_nodes * max_num_nodes + idx1 * max_num_nodes + idx2
    adj = unsorted_segment_sum(edge_attr, idx)
    adj = tlx.reshape(adj, size)

    return adj