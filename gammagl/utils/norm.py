import tensorlayerx as tlx
import gammagl.mpops as mpops

def calc_gcn_norm(edge_index, num_nodes, edge_weight=None):
    """
    calculate GCN normilization.
    Since tf not support update value of a Tensor, we use np for calculation on CPU device.
    Args:
        edge_index: edge index
        num_nodes: number of nodes of graph
        edge_weight: edge weights of graph

    Returns:
        1-dim Tensor
    """
    src, dst = edge_index[0], edge_index[1]
    if edge_weight is None:
        edge_weight = tlx.ones((edge_index.shape[1],)) # torch backend `shape` 参数不能是int
    deg = tlx.reshape(mpops.unsorted_segment_sum(tlx.reshape(edge_weight,(-1,1)), src, num_segments=num_nodes), (-1,))# tlx更新后可以去掉 reshape
    deg_inv_sqrt = tlx.pow(deg, -0.5)
    # deg_inv_sqrt[tlx.is_inf(deg_inv_sqrt)] = 0 # may exist solo node
    weights = tlx.gather(deg_inv_sqrt,src) * edge_weight * tlx.gather(deg_inv_sqrt,dst)
    return weights