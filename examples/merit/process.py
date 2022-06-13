import scipy.sparse as sp
import numpy as np
import tensorlayerx as tlx
def compute_diff(A: sp.csr_matrix, alpha: float, eps: float):
    N = A.shape[0]
    A_loop = sp.eye(N) + A
    D_loop_vec = A_loop.sum(0).A1
    D_loop_vec_invsqrt = 1 / np.sqrt(D_loop_vec)
    D_loop_invsqrt = sp.diags(D_loop_vec_invsqrt)
    T_sym = D_loop_invsqrt @ A_loop @ D_loop_invsqrt
    S = alpha * sp.linalg.inv(sp.eye(N) - (1 - alpha) * T_sym)
    S_tilde = S.multiply(S >= eps)
    D_tilde_vec = S_tilde.sum(0).A1
    
    diff = S_tilde / D_tilde_vec
    norm_diff = sp.csr_matrix(diff).tocoo()
    diff_row, diff_col,diff_weight = norm_diff.row,norm_diff.col,norm_diff.data
    diff_edge_index =tlx.convert_to_tensor([diff_col,diff_row],dtype='int64')
    diff_weight=tlx.convert_to_tensor(diff_weight,dtype='float32')
    return diff_edge_index,diff_weight
