import tensorlayerx as tlx
import numpy as np
from scipy.sparse import csr_matrix


def build_sim(context):
    context_norm = context.div(tlx.l2_normalize(context, axis=-1))
    sim = tlx.matmul(context_norm, context_norm.transpose(1, 0))
    return sim

def build_knn_normalized_graph(adj, topk, is_sparse, norm_type):
    device = adj.device
    knn_val, knn_ind = tlx.topk(adj, topk, dim=-1)  
    n_item = knn_val.shape[0]
    n_data = knn_val.shape[0]*knn_val.shape[1]
    data = np.ones(n_data)
    if is_sparse:
        tuple_list = [[row, int(col)] for row in range(len(knn_ind)) for col in knn_ind[row]]  #
        row = [i[0] for i in tuple_list]  #
        col = [i[1] for i in tuple_list]  #
        ii_graph = csr_matrix((data, (row, col)) ,shape=(n_item, n_item))
        return ii_graph
    else:
        weighted_adjacency_matrix = (tlx.zeros_like(adj)).scatter_(-1, knn_ind, knn_val)
        return get_dense_laplacian(weighted_adjacency_matrix, normalization=norm_type)


def get_sparse_laplacian(edge_index, edge_weight, num_nodes, normalization='none'):  
    row, col = edge_index[0], edge_index[1]  
    deg = tlx.unsorted_segment_sum(edge_weight, row, num_nodes)  
    

    if normalization == 'sym':
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    elif normalization == 'rw':
        deg_inv = 1.0 / deg
        deg_inv.masked_fill_(deg_inv == float('inf'), 0)
        edge_weight = deg_inv[row] * edge_weight
    return edge_index, edge_weight


def get_dense_laplacian(adj, normalization='none'):
    if normalization == 'sym':
        rowsum = tlx.reduce_sum(adj, -1)
        d_inv_sqrt = tlx.pow(rowsum, -0.5)
        d_inv_sqrt[tlx.is_inf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = tlx.diag(d_inv_sqrt)
        L_norm = tlx.matmul(tlx.matmul(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
    elif normalization == 'rw':
        rowsum =  tlx.reduce_sum(adj, -1)
        d_inv = tlx.pow(rowsum, -1)
        d_inv[tlx.is_inf(d_inv)] = 0.
        d_mat_inv = tlx.diag(d_inv)
        L_norm = tlx.matmul(d_mat_inv, adj)
    elif normalization == 'none':
        L_norm = adj
    return L_norm
