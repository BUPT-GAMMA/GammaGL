# coding=utf-8

import tensorflow as tf
from gammagl.sparse.sparse_adj import CSRAdj


# sparse_adj @ diagonal_matrix
def sparse_diag_matmul(sparse_adj: CSRAdj, diagonal):
    return sparse_adj.matmul_diag(diagonal)


# self @ diagonal_matrix
def diag_sparse_matmul(diagonal, sparse_adj: CSRAdj):
    return sparse_adj.rmatmul_diag(diagonal)


