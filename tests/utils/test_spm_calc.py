import numpy as np
import scipy.sparse as sp
from gammagl.utils.spm_calc import calc_A_norm_hat


def test_calc_A_norm_hat():
    edge_index = np.array([
        [0, 1, 1, 2],
        [1, 0, 2, 1]
    ])
    A = np.array([
        [1, 1, 0],  
        [1, 1, 1], 
        [0, 1, 1]  
    ])
    D_vec = np.sum(A, axis=1)
    D_vec_invsqrt_corr = 1 / np.sqrt(D_vec)
    D_invsqrt_corr = np.diag(D_vec_invsqrt_corr)
    expected_output = D_invsqrt_corr @ A @ D_invsqrt_corr
    result = calc_A_norm_hat(edge_index).toarray()
    assert np.allclose(result, expected_output)
