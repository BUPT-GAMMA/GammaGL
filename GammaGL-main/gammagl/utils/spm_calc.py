import numpy as np
import scipy.sparse as sp

def calc_A_norm_hat(edge_index, weights=None):
    if weights is None:
        weights = np.ones(edge_index.shape[1])
    sparse_adj = sp.coo_matrix((weights, (edge_index[0], edge_index[1])))
    nnodes = sparse_adj.shape[0]
    A = sparse_adj + sp.eye(nnodes)
    D_vec = np.sum(A, axis=1).A1
    D_vec_invsqrt_corr = 1 / np.sqrt(D_vec)
    D_invsqrt_corr = sp.diags(D_vec_invsqrt_corr)
    return D_invsqrt_corr @ A @ D_invsqrt_corr
