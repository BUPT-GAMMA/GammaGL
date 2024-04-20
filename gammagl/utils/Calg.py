import tensorlayerx as tlx
from gammagl.mpops import *

def cal_g_gradient(edge_index, x, edge_weight=None, sigma1=0.5, sigma2=0.5, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):
    row, col = edge_index[0], edge_index[1]
    ones = tlx.ones([edge_index[0].shape[0]])
    if dtype is not None:
        ones = tlx.cast(ones, dtype)
    if num_nodes is None:
        num_nodes = int(1 + tlx.reduce_max(edge_index[0]))
    deg=unsorted_segment_sum(ones, col, num_segments=num_nodes)
    deg_inv = tlx.pow(deg+1e-8, -1)
    deg_in_row=tlx.reshape(tlx.gather(deg_inv,row),(-1,1))
    x_row=tlx.gather(x,row)
    x_col=tlx.gather(x,col)
    gra = deg_in_row * (x_col - x_row)
    avg_gra=unsorted_segment_sum(gra, row, num_segments=x.shape[0])
    dx = x_row-x_col
    norms_dx = tlx.reduce_sum(tlx.square(dx), axis=1)
    norms_dx = tlx.sqrt(norms_dx)
    s = norms_dx
    s = tlx.exp(- (s * s) / (2 * sigma1 * sigma2))
    r=unsorted_segment_sum(tlx.reshape(s,(-1,1)),row,num_segments=x.shape[0])
    r_row=tlx.gather(r,row)
    coe = tlx.reshape(s,(-1,1)) / (r_row + 1e-6)
    avg_gra_row=tlx.gather(avg_gra,row)
    result=unsorted_segment_sum(avg_gra_row * coe, col, num_segments=x.shape[0])
    return result