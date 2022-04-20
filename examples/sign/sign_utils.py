import numpy as np
import scipy.sparse as sp


def calc_sign(adj, deg, feat):
    col = adj[0]
    row = adj[1]
    deg_inv_sqrt = np.power(deg, -0.5).flatten()

    weight = np.ones_like(col)
    new_weight = deg_inv_sqrt[row] * weight * deg_inv_sqrt[col]
    new_adj = sp.coo_matrix((new_weight, [col, row]))
    return new_adj @ feat



def normalize_feat(feat):
    feat = feat - np.min(feat)
    #
    feat = np.divide(feat, feat.sum(axis=-1, keepdims=True).clip(min=1.))
    return feat