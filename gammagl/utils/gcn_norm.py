import tensorlayerx as tlx
from gammagl.mpops import *
from gammagl.utils.loop import remove_self_loops, add_self_loops,contains_self_loops
def gcn_norm(edge_index, num_nodes, edge_weight=None,add_self_loop=False):
   
    if edge_weight is None:
        edge_weight = tlx.ones(shape=(edge_index.shape[1], 1))  
    if add_self_loop:
        if contains_self_loops(edge_index):
            edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
            edge_index,edge_weight= add_self_loops(edge_index, edge_weight)
        else:
            edge_index,edge_weight= add_self_loops(edge_index, edge_weight)
    src, dst = edge_index[0], edge_index[1]


    deg = tlx.reshape(unsorted_segment_sum(edge_weight,src , num_segments=edge_index[0].shape[0]), (-1,))
    deg_inv_sqrt = tlx.pow(deg+1e-8, -0.5)
    weights = tlx.gather(deg_inv_sqrt, src) * tlx.reshape(edge_weight, (-1,)) * tlx.gather(deg_inv_sqrt, dst)
    return edge_index,weights